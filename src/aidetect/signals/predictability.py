"""Predictability backends ("perplexity").

``unigram``   Surprisal of each word under the wordfreq corpus frequency
              distribution (-log2 p(word)). It needs no neural network and is
              what the shipped "lite" model uses. It measures how common the
              chosen words are, *not* how predictable they are in context, so it
              is a weak proxy for language-model perplexity.
``causal_lm`` True per-token surprisal under a causal language model such as
              GPT-2 (optional "full" profile; needs torch + transformers and the
              model weights, downloaded once and cached locally).

Why perplexity is never used alone: low perplexity means "the text is easy for a
language model to predict". Generated text tends to be low-perplexity because
decoding favours probable tokens, but polished human prose, formulaic genres
(legal, scientific, news), simple vocabulary and memorised text are low-
perplexity too. Conversely, high-temperature sampling or paraphrasing raises the
perplexity of AI text. Perplexity therefore enters the ensemble as one signal
among several, and its variation (burstiness) is measured separately.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from ..lexicons import fold
from ..preprocessing import Document

FREQ_FLOOR = 1e-8  # Zipf 1.0; assigned to words missing from the frequency list


@lru_cache(maxsize=4)
def folded_frequencies(lang: str) -> dict[str, float]:
    """wordfreq frequencies keyed by folded (accent-free, lower-case) word."""
    from wordfreq import get_frequency_dict

    out: dict[str, float] = {}
    for w, f in get_frequency_dict(lang, wordlist="best").items():
        k = fold(w)
        out[k] = out.get(k, 0.0) + f
    return out


def zipf(word: str, freqs: dict[str, float]) -> float:
    f = freqs.get(word, 0.0)
    return math.log10(f) + 9.0 if f > 0 else 0.0


class UnigramSurprisal:
    name = "unigram"
    unit = "bits/word"

    def __init__(self, lang: str):
        self.lang = lang
        self.freqs = folded_frequencies(lang)
        self._cache: dict[str, float] = {}

    def surprisal(self, word: str) -> float:
        v = self._cache.get(word)
        if v is None:
            v = -math.log2(max(self.freqs.get(word, 0.0), FREQ_FLOOR))
            self._cache[word] = v
        return v

    def annotate(self, doc: Document) -> None:
        for s in doc.sentences:
            s.surprisal = np.fromiter((self.surprisal(w) for w in s.words), dtype=np.float64, count=len(s.words))


class CausalLMSurprisal:
    """Per-token surprisal (bits) from a causal LM, mapped back onto sentences."""

    name = "causal_lm"
    unit = "bits/token"

    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 1024, stride: int = 512):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The 'full' profile needs torch and transformers (pip install -r requirements-optional.txt)."
            ) from exc
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
        self.device = device
        ctx = getattr(self.model.config, "n_positions", None) or getattr(self.model.config, "max_position_embeddings", 1024)
        self.max_length = min(max_length, int(ctx))
        self.stride = min(stride, self.max_length // 2)
        self.model_name = model_name

    def token_surprisal(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (surprisal_bits[n_tokens], char_offsets[n_tokens, 2])."""
        torch = self.torch
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids = enc["input_ids"]
        offsets = np.asarray(enc["offset_mapping"], dtype=np.int64).reshape(-1, 2)
        n = len(ids)
        bits = np.full(n, np.nan)
        if n < 2:
            return bits, offsets
        bos = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id
        start = 0
        while start < n:
            step = self.max_length - 1 if start == 0 else self.stride
            end = min(start + step, n)
            ctx_start = max(0, end - (self.max_length - 1))
            chunk = ids[ctx_start:end]
            inp = torch.tensor([[bos] + chunk], device=self.device)
            with torch.no_grad():
                logits = self.model(inp).logits[0, :-1]
                logp = torch.log_softmax(logits.float(), dim=-1)
                tgt = torch.tensor(chunk, device=self.device)
                nll = -logp[torch.arange(len(chunk), device=self.device), tgt].cpu().numpy() / math.log(2)
            for k in range(start, end):
                bits[k] = nll[k - ctx_start]
            if end == n:
                break
            start = end
        return bits, offsets

    def annotate(self, doc: Document) -> None:
        bits, offs = self.token_surprisal(doc.text)
        mids = offs.mean(axis=1) if len(offs) else np.zeros(0)
        starts = np.array([s.start for s in doc.sentences])
        ends = np.array([s.end for s in doc.sentences])
        for i, s in enumerate(doc.sentences):
            mask = (mids >= starts[i]) & (mids < ends[i]) & ~np.isnan(bits)
            s.surprisal = bits[mask]


def make_predictability_backend(kind: str, lang: str, config: dict | None = None, device: str = "cpu"):
    if kind == "unigram":
        return UnigramSurprisal(lang)
    if kind == "causal_lm":
        names = ((config or {}).get("profiles", {}).get("full", {}).get("causal_lm", {}))
        name = names.get(lang)
        if not name:
            raise RuntimeError(f"No causal language model configured for language '{lang}' (profiles.full.causal_lm).")
        return CausalLMSurprisal(name, device=device)
    raise ValueError(f"Unknown predictability backend: {kind}")
