"""Controlled multi-model synthetic data generation.

Produces AI-written counterparts of human documents (topic-matched, so the
detector cannot learn topic instead of authorship) from several model families,
plus the borderline categories the training data needs:

    continue        write a new document on the same topic/genre as a human source
    paraphrase      rewrite a human text (AI-paraphrased human text)
    paraphrase_ai   rewrite an AI text (paraphrased AI text)
    polish          correct grammar/style of a human text (AI-assisted human text)
    translate_el    translate into Greek (machine-translated text)
    backtranslate   English -> Greek -> English (back-translation)
    write_el        write a Greek document on the source topic

Backends
    anthropic          Claude models via the official ``anthropic`` SDK (pip install anthropic)
    openai_compatible  any OpenAI-compatible chat endpoint (OpenAI, Mistral, vLLM, LM Studio ...)
    gemini             Google Gemini REST API
    ollama             local models (Llama, Mistral, Qwen ...) via Ollama - fully local

Cloud backends upload the source text to a third party. They only run with the
explicit ``--allow-external-upload`` flag; the local Ollama backend does not need it.
Every sample records the model that actually produced it (for Claude, the model
named in the response, which differs from the requested one when a refusal
fallback served the request).

    python -m aidetect.generation --input data/processed/en/train.jsonl --backend ollama --model llama3.1 \\
        --tasks continue,paraphrase --n 200 --out data/generated/ollama_llama31.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from .data import clean_text, load_jsonl, make_sample, save_jsonl, sha1

PROMPTS = {
    "continue": ("Write a {genre} text of roughly {words} words on the same topic as the following excerpt. "
                 "Write only the text itself, without a title or any commentary.\n\nExcerpt:\n{excerpt}"),
    "paraphrase": "Rewrite the following text in your own words, keeping its meaning and length. Output only the rewritten text.\n\n{text}",
    "paraphrase_ai": "Paraphrase the following text thoroughly, changing wording and sentence structure. Output only the paraphrase.\n\n{text}",
    "polish": "Correct the grammar, spelling and style of the following text without changing its content. Output only the corrected text.\n\n{text}",
    "translate_el": "Translate the following text into Greek. Output only the translation.\n\n{text}",
    "to_en": "Translate the following text into English. Output only the translation.\n\n{text}",
    "write_el": ("Γράψε ένα κείμενο ({genre}) περίπου {words} λέξεων στα ελληνικά, με το ίδιο θέμα με το παρακάτω απόσπασμα. "
                 "Γράψε μόνο το κείμενο, χωρίς τίτλο ή σχόλια.\n\nΑπόσπασμα:\n{excerpt}"),
}
TASK_META = {  # task -> (label, author_type, editing_level, paraphrasing_level, language)
    "continue": (1, "ai", "none", "none", "en"),
    "paraphrase": (1, "ai_paraphrased_human", "none", "model", "en"),
    "paraphrase_ai": (1, "paraphrased_ai", "none", "model", "en"),
    "polish": (1, "ai_assisted_human", "ai_polished", "none", "en"),
    "translate_el": (1, "machine_translated", "none", "none", "el"),
    "backtranslate": (1, "machine_translated", "none", "back_translation", "en"),
    "write_el": (1, "ai", "none", "none", "el"),
}
CLAUDE_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}


@dataclass
class Generation:
    text: str
    model: str          # model that actually produced the text
    finish: str


class GenerationRefused(Exception):
    pass


# --------------------------------------------------------------------------- backends
class AnthropicBackend:
    """Claude via the official SDK. Sampling temperature is not configurable on current
    Claude models (the API rejects it), so diversity comes from prompts and sources."""

    family = "anthropic"
    external = True

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", client=None):
        if client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise RuntimeError("Install the Anthropic SDK first: pip install anthropic") from exc
            client = anthropic.Anthropic()  # credentials from ANTHROPIC_API_KEY or an `ant auth login` profile
        self.client = client
        self.model = model
        self.effort = effort

    def generate(self, prompt: str, temperature: float | None = None, max_words: int = 800) -> Generation:
        kwargs = dict(model=self.model, max_tokens=16000, output_config={"effort": self.effort},
                      messages=[{"role": "user", "content": prompt}])
        if self.model in CLAUDE_FALLBACK_MODELS:
            # Server-side refusal fallback: a declined request is re-run on a fallback model in the same call.
            resp = self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        else:
            resp = self.client.messages.create(**kwargs)
        if resp.stop_reason == "refusal":
            raise GenerationRefused(getattr(resp, "stop_details", None))
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return Generation(text=text, model=resp.model, finish=str(resp.stop_reason))


class OpenAICompatibleBackend:
    external = True

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1", api_key_env: str = "OPENAI_API_KEY",
                 family: str = "openai", transport: httpx.BaseTransport | None = None):
        if not model:
            raise ValueError("--model is required for the openai_compatible backend")
        self.model, self.family = model, family
        key = os.environ.get(api_key_env, "")
        self.http = httpx.Client(base_url=base_url, timeout=300, transport=transport,
                                 headers={"Authorization": f"Bearer {key}"} if key else {})

    def generate(self, prompt: str, temperature: float | None = 0.7, max_words: int = 800) -> Generation:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": int(max_words * 2.5)}
        if temperature is not None:
            body["temperature"] = temperature
        r = self.http.post("/chat/completions", json=body)
        r.raise_for_status()
        data = r.json()
        ch = data["choices"][0]
        return Generation(text=ch["message"]["content"] or "", model=data.get("model", self.model),
                          finish=str(ch.get("finish_reason")))


class GeminiBackend:
    family = "google"
    external = True

    def __init__(self, model: str, api_key_env: str = "GEMINI_API_KEY", transport: httpx.BaseTransport | None = None):
        if not model:
            raise ValueError("--model is required for the gemini backend")
        self.model = model
        self.key = os.environ.get(api_key_env, "")
        self.http = httpx.Client(base_url="https://generativelanguage.googleapis.com/v1beta", timeout=300, transport=transport)

    def generate(self, prompt: str, temperature: float | None = 0.7, max_words: int = 800) -> Generation:
        body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {}}
        if temperature is not None:
            body["generationConfig"]["temperature"] = temperature
        r = self.http.post(f"/models/{self.model}:generateContent", params={"key": self.key}, json=body)
        r.raise_for_status()
        data = r.json()
        cand = (data.get("candidates") or [{}])[0]
        parts = cand.get("content", {}).get("parts", [])
        return Generation(text="".join(p.get("text", "") for p in parts), model=self.model,
                          finish=str(cand.get("finishReason")))


class OllamaBackend:
    external = False

    def __init__(self, model: str, base_url: str = "http://localhost:11434", family: str | None = None,
                 transport: httpx.BaseTransport | None = None):
        if not model:
            raise ValueError("--model is required for the ollama backend")
        self.model = model
        low = model.lower()
        self.family = family or next((f for f in ("llama", "mistral", "qwen", "gemma", "phi", "deepseek") if f in low), "open-source")
        self.http = httpx.Client(base_url=base_url, timeout=600, transport=transport)

    def generate(self, prompt: str, temperature: float | None = 0.7, max_words: int = 800) -> Generation:
        body = {"model": self.model, "prompt": prompt, "stream": False, "options": {"num_predict": int(max_words * 2.5)}}
        if temperature is not None:
            body["options"]["temperature"] = temperature
        r = self.http.post("/api/generate", json=body)
        r.raise_for_status()
        data = r.json()
        return Generation(text=data.get("response", ""), model=data.get("model", self.model),
                          finish=str(data.get("done_reason", "stop")))


def make_backend(name: str, model: str | None, **kw):
    if name == "anthropic":
        return AnthropicBackend(model or "claude-opus-5-5", effort=kw.get("effort", "medium"))
    if name == "openai_compatible":
        return OpenAICompatibleBackend(model, base_url=kw.get("base_url") or "https://api.openai.com/v1",
                                       api_key_env=kw.get("api_key_env") or "OPENAI_API_KEY",
                                       family=kw.get("family") or "openai")
    if name == "gemini":
        return GeminiBackend(model)
    if name == "ollama":
        return OllamaBackend(model, base_url=kw.get("base_url") or "http://localhost:11434", family=kw.get("family"))
    raise ValueError(f"Unknown backend {name}")


# --------------------------------------------------------------------------- generation loop
def _excerpt(text: str, words: int = 60) -> str:
    return " ".join(text.split()[:words])


def run_task(backend, task: str, source: dict, temperature: float | None, rng: random.Random) -> dict | None:
    words = max(150, min(700, source["n_words"]))
    genre = source.get("genre", "general")
    if task == "backtranslate":
        g1 = backend.generate(PROMPTS["translate_el"].format(text=source["text"]), temperature, words)
        g = backend.generate(PROMPTS["to_en"].format(text=g1.text), temperature, words)
    elif task in ("continue", "write_el"):
        g = backend.generate(PROMPTS[task].format(genre=genre, words=words, excerpt=_excerpt(source["text"])), temperature, words)
    else:
        g = backend.generate(PROMPTS[task].format(text=source["text"]), temperature, words)
    text = clean_text(g.text, strip_labels=True)
    if len(text.split()) < 40:
        return None
    label, atype, edit, para, lang = TASK_META[task]
    s = make_sample(text, label, source_dataset="generated", domain=source["domain"], generator=g.model,
                    generator_family=backend.family, model_version=g.model, author_type=atype,
                    prompt_type=task, group_id=source["group_id"], language=lang,
                    generation_date=time.strftime("%Y-%m-%d"), editing_level=edit, paraphrasing_level=para,
                    extra={"source_id": source["id"], "temperature": temperature, "finish_reason": g.finish})
    return s


def generate_dataset(backend, sources: list[dict], tasks: list[str], n: int, temperatures: list[float | None],
                     seed: int = 0, progress: bool = True) -> list[dict]:
    rng = random.Random(seed)
    pool = list(sources)
    rng.shuffle(pool)
    out = []
    for i, src in enumerate(pool[:n]):
        task = tasks[i % len(tasks)]
        if task in ("paraphrase", "polish", "translate_el", "backtranslate", "continue", "write_el") and src["label"] != 0:
            continue
        if task == "paraphrase_ai" and src["label"] != 1:
            continue
        temp = temperatures[i % len(temperatures)]
        try:
            s = run_task(backend, task, src, temp, rng)
        except GenerationRefused as exc:
            print(f"  refused ({exc}); skipped", file=sys.stderr)
            continue
        except httpx.HTTPError as exc:
            print(f"  request failed: {exc}; skipped", file=sys.stderr)
            continue
        if s:
            out.append(s)
        if progress and (i + 1) % 20 == 0:
            print(f"  {i + 1}/{min(n, len(pool))} processed, {len(out)} kept", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="jsonl of source documents (with metadata)")
    ap.add_argument("--backend", required=True, choices=["anthropic", "openai_compatible", "gemini", "ollama"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--family", default=None, help="model family label used for leave-one-family-out tests")
    ap.add_argument("--tasks", default="continue")
    ap.add_argument("--temperatures", default="0.7", help="comma list; use 'none' for backends without temperature")
    ap.add_argument("--effort", default="medium", help="Claude effort level (low|medium|high|xhigh|max)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-external-upload", action="store_true",
                    help="required for cloud backends: source texts are sent to the provider")
    args = ap.parse_args(argv)
    backend = make_backend(args.backend, args.model, base_url=args.base_url, api_key_env=args.api_key_env,
                           family=args.family, effort=args.effort)
    if backend.external and not args.allow_external_upload:
        print("This backend sends the source texts to an external provider. Re-run with --allow-external-upload "
              "if that is acceptable for these texts.")
        return 2
    temps = [None if t.strip().lower() == "none" else float(t) for t in args.temperatures.split(",")]
    if args.backend == "anthropic":
        temps = [None]
    sources = load_jsonl(args.input)
    out = generate_dataset(backend, sources, args.tasks.split(","), args.n, temps, args.seed)
    save_jsonl(out, args.out)
    print(f"Wrote {len(out)} samples to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
