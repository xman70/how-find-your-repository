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
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

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
    if a.cli:
        return _cli(a)
    if a.daemon:
        return _daemon(a)
    return _gui()


if __name__ == "__main__":
    sys.exit(main())
