"""VUSA AI Quant Terminal - entry point.

Usage
-----
    python main.py                       # GUI
    python main.py --cli                 # headless analysis (balanced mode)
    python main.py --cli --mode deep     # headless deep research
    python main.py --cli --demo          # synthetic demo data (clearly labelled)
    python main.py --daemon              # headless daily scheduler (trading days only)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ESSENTIAL = ["numpy", "pandas", "scipy", "sklearn", "statsmodels", "plotly", "requests"]


def _use_project_venv() -> None:
    """Started with a different Python (e.g. double-clicked main.py)? Re-run with the environment
    that install.bat / bootstrap.py created - but only if that installation finished."""
    import subprocess

    if os.environ.get("VUSA_NO_REEXEC"):
        return
    sys.path.insert(0, str(ROOT))
    try:
        import bootstrap
    except Exception:  # noqa: BLE001
        return
    finally:
        sys.path.remove(str(ROOT))
    venv = bootstrap.existing_venv()
    if venv is None:
        return
    py = bootstrap.venv_python(venv)
    try:
        if Path(sys.prefix).resolve() == venv.resolve():  # already running inside that environment
            return
    except OSError:
        return
    rc = subprocess.call([str(py), str(Path(__file__).resolve()), *sys.argv[1:]],
                         env=dict(os.environ, VUSA_NO_REEXEC="1"))
    if rc != 0 and os.name == "nt" and sys.stdin and sys.stdin.isatty():
        input("\nThe application stopped with an error (see above). Press Enter to close...")
    sys.exit(rc)


def _register_package() -> None:
    """Make 'import vusa_quant_ai' work even if this folder was renamed (e.g. by unzipping)."""
    import importlib.util

    if ROOT.name == "vusa_quant_ai":
        if str(ROOT.parent) not in sys.path:
            sys.path.insert(0, str(ROOT.parent))
        return
    if "vusa_quant_ai" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location("vusa_quant_ai", ROOT / "__init__.py",
                                                  submodule_search_locations=[str(ROOT)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vusa_quant_ai"] = mod
    spec.loader.exec_module(mod)


def _missing_packages(gui: bool) -> list[str]:
    import importlib.util

    names = ESSENTIAL + (["PySide6"] if gui else [])
    return [n for n in names if importlib.util.find_spec(n) is None]


def _explain_missing(missing: list[str]) -> int:
    print("\n" + "=" * 72)
    print(" VUSA AI Quant Terminal cannot start: required packages are not installed.")
    print(f" Missing: {', '.join(missing)}")
    print(f" Python used: {sys.executable} ({sys.version.split()[0]})")
    print("\n How to fix: double-click  run.bat  in this folder. It installs everything")
    print(" automatically on the first start (this takes a few minutes), then opens the app.")
    print("=" * 72 + "\n")
    if os.name == "nt" and sys.stdin and sys.stdin.isatty():
        input("Press Enter to close...")
    return 3


for _stream in (sys.stdout, sys.stderr):  # Windows consoles (e.g. cp1253/cp437) cannot print emoji like 🟢
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

def _ascii_ca_bundle() -> None:
    """curl_cffi (used by yfinance) hands the CA-bundle path to libcurl in the ANSI code page, so a
    non-ASCII path (e.g. a Greek user name) makes every yfinance request fail on Windows. Point it
    at an ASCII copy of the certifi bundle instead."""
    if os.name != "nt" or any(os.environ.get(v) for v in ("SSL_CERT_FILE", "CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE")):
        return
    try:
        import shutil

        import certifi

        src = certifi.where()
        if src.isascii():
            return
        for base in (os.environ.get("ProgramData", ""), os.environ.get("SystemDrive", "C:") + "\\"):
            if not base or not base.isascii():
                continue
            try:
                dst = Path(base) / "VUSA-Quant" / "cacert.pem"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists() or dst.stat().st_size != Path(src).stat().st_size:
                    shutil.copyfile(src, dst)
                os.environ["CURL_CA_BUNDLE"] = str(dst)
                os.environ["SSL_CERT_FILE"] = str(dst)
                return
            except OSError:
                continue
    except Exception:  # noqa: BLE001 - never block start-up over this
        pass


_use_project_venv()
_register_package()

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*frequency information.*")
warnings.filterwarnings("ignore", module="statsmodels")


def _cli(args) -> int:
    from vusa_quant_ai.config.settings import DATA_DIR, load_settings
    from vusa_quant_ai.core.runtime import add_file_logging
    from vusa_quant_ai.services.api import Backend

    s = load_settings()
    add_file_logging(DATA_DIR / "logs")
    b = Backend(s)
    res = b.analyze(args.mode, args.offline, True if args.demo else None, args.as_of, args.llm,
                    progress=lambda m, f: print(f"{100 * f:5.1f}%  {m}", flush=True))
    if not res.ok:
        print("\nANALYSIS FAILED:", res.error)
        return 2
    print("\n" + res.state["report_markdown"])
    if args.backtest:
        bt = b.backtest(20, progress=lambda m, f: print(m, flush=True))
        print("\nBACKTEST:", bt.result.metrics, "\nBUY&HOLD:", bt.result.benchmark_metrics)
    return 0


def _daemon(args) -> int:
    from vusa_quant_ai.config.settings import load_settings
    from vusa_quant_ai.pipeline.scheduler import next_run_time, should_run_now
    from vusa_quant_ai.services.api import Backend

    s = load_settings()
    b = Backend(s)
    last = None
    print("Daily scheduler running (Ctrl+C to stop). Weekends and exchange holidays are skipped.")
    while True:
        now = datetime.now().astimezone()
        ok, why = should_run_now(now, last, s.schedule.daily_time, s.schedule.timezone, s.exchange)
        if ok:
            res = b.analyze(s.mode if s.mode != "continuous" else "balanced", progress=lambda m, f: print(m, flush=True))
            last = str(now.date())
            print(f"[{now:%Y-%m-%d %H:%M}] run {'ok' if res.ok else 'FAILED: ' + res.error}")
        else:
            nr = next_run_time(now, s.schedule.daily_time, s.schedule.timezone, s.exchange)
            print(f"[{now:%H:%M}] {why}; next run {nr:%a %Y-%m-%d %H:%M}", flush=True)
        time.sleep(300)


def _gui() -> int:
    from PySide6.QtWidgets import QApplication

    from vusa_quant_ai.config.settings import DATA_DIR, load_settings
    from vusa_quant_ai.core.runtime import add_file_logging
    from vusa_quant_ai.gui.main_window import MainWindow
    from vusa_quant_ai.gui.widgets import DARK_QSS

    os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-logging")
    if hasattr(os, "geteuid") and os.geteuid() == 0:  # Chromium refuses to sandbox as root (Linux only)
        os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")
    s = load_settings()
    add_file_logging(DATA_DIR / "logs")
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_QSS)
    w = MainWindow(s)
    w.show()
    return app.exec()


def main() -> int:
    p = argparse.ArgumentParser(description="VUSA AI Quant Terminal")
    p.add_argument("--cli", action="store_true", help="run one analysis without GUI")
    p.add_argument("--daemon", action="store_true", help="headless daily scheduler")
    p.add_argument("--mode", default=None, choices=["fast", "balanced", "deep", "continuous"])
    p.add_argument("--offline", action="store_true", help="use cached data only")
    p.add_argument("--demo", action="store_true", help="synthetic demo data (clearly labelled)")
    p.add_argument("--as-of", dest="as_of", default=None, help="historical as-of date (YYYY-MM-DD)")
    p.add_argument("--llm", action="store_true", help="LLM-phrased explanation (needs ANTHROPIC_API_KEY)")
    p.add_argument("--backtest", action="store_true", help="also run the walk-forward backtest (CLI)")
    a = p.parse_args()
    missing = _missing_packages(gui=not (a.cli or a.daemon))
    if missing:
        return _explain_missing(missing)
    _ascii_ca_bundle()
    if a.cli:
        return _cli(a)
    if a.daemon:
        return _daemon(a)
    return _gui()


if __name__ == "__main__":
    sys.exit(main())
