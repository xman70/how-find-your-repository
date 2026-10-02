"""FastAPI application (local web UI + JSON API).

Binds to 127.0.0.1 only. Documents are processed on this computer; nothing is
uploaded anywhere. Run with ``python app.py`` (or START.bat on Windows).
"""
from __future__ import annotations

import os
import socket
import threading
import webbrowser
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import DISCLAIMER, PRIVACY_NOTE, __version__
from .config import load_config, resolve_path
from .io_utils import DocumentReadError, read_bytes
from .plagiarism import SimilarityChecker
from .predict import Analyzer
from .report import to_html, to_json, to_pdf
from .storage import Store

WEB_DIR = Path(__file__).parent / "web"


class AnalyzeRequest(BaseModel):
    text: str = Field(..., max_length=2_000_000)
    title: str | None = None
    language: str | None = None


class TextRequest(BaseModel):
    text: str = Field(..., max_length=2_000_000)
    title: str | None = None


class SettingsRequest(BaseModel):
    device: str = Field("auto", pattern="^(auto|cpu|gpu)$")


def create_app(config: dict | None = None, analyzer: Analyzer | None = None, store: Store | None = None) -> FastAPI:
    cfg = config or load_config()
    store = store or Store(resolve_path(cfg["app"]["database"]))
    state = {"analyzer": analyzer}
    max_bytes = int(cfg["app"]["max_upload_mb"]) * 1024 * 1024

    def get_analyzer() -> Analyzer:
        if state["analyzer"] is None:
            state["analyzer"] = Analyzer(cfg, device=store.get_setting("device", cfg["runtime"]["device"]))
        return state["analyzer"]

    checker = SimilarityChecker(store, cfg)
    app = FastAPI(title="AI Text Analysis", version=__version__, docs_url="/api/docs", redoc_url=None)
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (WEB_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/vendor/plotly.min.js")
    def plotly_js():
        import plotly

        p = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
        if not p.exists():
            raise HTTPException(404, "plotly.min.js not found in the installed plotly package")
        return FileResponse(p, media_type="application/javascript")

    @app.get("/api/health")
    def health():
        an = get_analyzer()
        return {"status": "ok", "version": __version__, "privacy": PRIVACY_NOTE, "disclaimer": DISCLAIMER,
                "device": an.device, "device_request": an.device_request, "device_note": an.device_note,
                "models": an.available_languages(), "supported_languages": cfg["languages"]["supported"]}

    def _finish(text: str, result: dict) -> dict:
        try:
            result["id"] = store.save_analysis(text, result, bool(cfg["app"]["store_text_in_history"]))
        except Exception as exc:  # noqa: BLE001 - history is optional
            result.setdefault("warnings", []).append(f"Could not save to history: {exc}")
        return result

    @app.post("/api/analyze")
    def analyze(req: AnalyzeRequest):
        res = get_analyzer().analyze(req.text, title=req.title, language=req.language)
        return _finish(req.text, res)

    @app.post("/api/analyze-file")
    async def analyze_file(file: UploadFile = File(...), title: str | None = Form(None)):
        data = await file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(413, f"File larger than {cfg['app']['max_upload_mb']} MB")
        try:
            text = read_bytes(data, file.filename or "upload.txt")
        except DocumentReadError as exc:
            raise HTTPException(400, str(exc)) from exc
        res = get_analyzer().analyze(text, title=title or file.filename)
        return _finish(text, res)

    @app.get("/api/analyses")
    def list_analyses(limit: int = 100):
        return store.list_analyses(limit)

    @app.get("/api/analyses/{analysis_id}")
    def get_analysis(analysis_id: int):
        r = store.get_analysis(analysis_id)
        if r is None:
            raise HTTPException(404, "Analysis not found")
        r["id"] = analysis_id
        return r

    @app.delete("/api/analyses/{analysis_id}")
    def delete_analysis(analysis_id: int):
        if not store.delete_analysis(analysis_id):
            raise HTTPException(404, "Analysis not found")
        return {"deleted": analysis_id}

    @app.get("/api/analyses/{analysis_id}/report.{fmt}")
    def report(analysis_id: int, fmt: str):
        r = store.get_analysis(analysis_id)
        if r is None:
            raise HTTPException(404, "Analysis not found")
        name = f"analysis-{analysis_id}"
        if fmt == "json":
            return Response(to_json(r), media_type="application/json",
                            headers={"Content-Disposition": f'attachment; filename="{name}.json"'})
        if fmt == "html":
            return Response(to_html(r), media_type="text/html",
                            headers={"Content-Disposition": f'attachment; filename="{name}.html"'})
        if fmt == "pdf":
            return Response(to_pdf(r), media_type="application/pdf",
                            headers={"Content-Disposition": f'attachment; filename="{name}.pdf"'})
        raise HTTPException(400, "Format must be json, html or pdf")

    @app.post("/api/similarity")
    def similarity(req: TextRequest):
        return checker.check(req.text)

    @app.get("/api/references")
    def references():
        return store.list_references()

    @app.post("/api/references")
    def add_reference(req: TextRequest):
        doc_id = checker.add_reference(req.title or "Untitled reference", req.text)
        return {"id": doc_id, "duplicate": doc_id is None}

    @app.post("/api/references/upload")
    async def add_reference_file(file: UploadFile = File(...)):
        data = await file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(413, "File too large")
        try:
            text = read_bytes(data, file.filename or "reference.txt")
        except DocumentReadError as exc:
            raise HTTPException(400, str(exc)) from exc
        doc_id = checker.add_reference(file.filename or "reference", text)
        return {"id": doc_id, "duplicate": doc_id is None}

    @app.delete("/api/references/{doc_id}")
    def delete_reference(doc_id: int):
        if not store.delete_reference(doc_id):
            raise HTTPException(404, "Reference not found")
        return {"deleted": doc_id}

    @app.get("/api/settings")
    def get_settings():
        an = get_analyzer()
        return {"device": an.device_request, "resolved_device": an.device, "note": an.device_note}

    @app.post("/api/settings")
    def set_settings(req: SettingsRequest):
        store.set_setting("device", req.device)
        state["analyzer"] = Analyzer(cfg, device=req.device)
        return get_settings()

    @app.exception_handler(Exception)
    async def unhandled(_, exc):  # pragma: no cover - last-resort handler
        return JSONResponse(status_code=500, content={"status": "error", "message": f"Internal error: {exc}"})

    return app


def _free_port(host: str, port: int) -> int:
    for p in range(port, port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    raise RuntimeError("No free port found")


@lru_cache(maxsize=1)
def app_instance() -> FastAPI:
    return create_app()


def run(open_browser: bool | None = None) -> None:
    import uvicorn

    cfg = load_config()
    host = cfg["app"]["host"]
    port = _free_port(host, int(cfg["app"]["port"]))
    url = f"http://{host}:{port}/"
    print(f"AI Text Analysis {__version__} - {PRIVACY_NOTE}\nOpen {url} in your browser (Ctrl+C to stop).", flush=True)
    if (cfg["app"]["open_browser"] if open_browser is None else open_browser) and not os.environ.get("AIDETECT_NO_BROWSER"):
        threading.Timer(2.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(cfg), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    run()
