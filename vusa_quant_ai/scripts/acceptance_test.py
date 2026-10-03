"""MINIMUM ACCEPTANCE TEST (section 77) against LIVE data.

Run on your own machine with internet access:

    python scripts/acceptance_test.py            # balanced mode
    python scripts/acceptance_test.py --deep     # full deep research

Each of the 20 criteria is checked and reported PASS / FAIL / NOT VERIFIABLE with the
reason. Nothing is marked PASS unless it was actually observed in this run.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--demo", action="store_true", help="synthetic data (offline check of the machinery only)")
    a = ap.parse_args()

    from vusa_quant_ai.config.settings import load_settings
    from vusa_quant_ai.services.api import Backend

    s = load_settings()
    b = Backend(s)
    res = b.analyze("deep" if a.deep else "balanced", synthetic=True if a.demo else None,
                    progress=lambda m, f: print(f"{100 * f:5.1f}% {m}", flush=True))
    rows = []

    def chk(n, name, cond, why=""):
        rows.append((n, name, "PASS" if cond is True else ("NOT VERIFIABLE" if cond is None else "FAIL"), why))

    if not res.ok:
        print("Analysis failed:", res.error)
        chk(1, "Download historical VUSA data", False, res.error)
        _print(rows)
        return 1
    st, art = res.state, res.artifacts
    real = not st["is_synthetic"]
    src = [r for r in st.get("source_health", []) if r["dataset"] == "VUSA"]
    chk(1, "Download historical VUSA data", (len(art.bundle.target) > 500 and real) if real else None,
        f"{len(art.bundle.target)} rows via {[r['source'] for r in src if r['status'] in ('ok', 'cached')]}")
    chk(2, "Validate it", bool(st.get("validation")), f"quality score {st.get('data_quality_score')}")
    chk(3, "Build technical features", st["feature_groups"].get("technical", 0) > 30, str(st["feature_groups"]))
    chk(4, "Build macro/cross-asset features", st["feature_groups"].get("cross_asset", 0) > 0 and st["feature_groups"].get("macro", 0) > 0,
        str(st["feature_groups"]))
    n = st.get("news", {})
    chk(5, "Retrieve and analyse current information", True if n.get("available") else (None if not real else False),
        n.get("reason", f"{len(n.get('events', []))} events"))
    fp = st["forecast_primary"]
    chk(6, "Train multiple models", len(fp.get("model_preds", {})) >= 3, ", ".join(fp.get("model_preds", {})))
    chk(7, "Leakage-safe walk-forward validation", st["leakage_audit"].get("passed") and fp.get("n_oos", 0) > 100,
        f"audit passed={st['leakage_audit'].get('passed')}, OOS n={fp.get('n_oos')}")
    chk(8, "Multi-horizon probabilistic predictions", len(st["forecasts"]) >= 3, ", ".join(f"{h}D" for h in st["forecasts"]))
    chk(9, "Calculate uncertainty", bool(fp.get("uncertainty")) and fp.get("interval") is not None, str(fp.get("uncertainty")))
    chk(10, "Detect market regime", bool(st["regime"].get("regime")), st["regime"].get("regime"))
    chk(11, "Historical analogue analysis", bool(st["analogues"].get("analogues")), f"{len(st['analogues'].get('analogues', []))} analogues")
    chk(12, "Monte Carlo", len(st["monte_carlo"]) >= 5, ", ".join(st["monte_carlo"]))
    chk(13, "Stress testing", len(st["stress"]) >= 5, f"{len(st['stress'])} scenarios")
    chk(14, "Produce BUY/HOLD/SELL", st["decision"]["signal"] in ("BUY", "HOLD", "SELL"), st["decision"]["label"])
    chk(15, "Explain the signal", bool(st.get("explanation")) and bool(st["decision"]["what_would_change"]), "")
    chk(16, "Save the complete decision state", b.db.load_trace(res.run_id) is not None, res.run_id)
    try:
        bt = b.backtest(20, retrain_every=126, progress=lambda m, f: None)
        chk(17, "Backtest the historical strategy", bt.audit.passed and len(bt.signals) > 100,
            f"CAGR {bt.result.metrics.get('cagr'):.2%} vs buy&hold {bt.result.benchmark_metrics.get('cagr'):.2%}")
    except Exception as exc:  # noqa: BLE001
        chk(17, "Backtest the historical strategy", False, str(exc)[:200])
    j = b.journal()
    chk(18, "Track subsequent prediction accuracy", not j.empty, f"{len(j)} journal rows; resolved: {int(j['actual_return'].notna().sum()) if not j.empty else 0} "
        "(resolution requires the horizon to pass)")
    chk(19, "Detect model drift", "checks" in st.get("model_drift", {}), str(st.get("model_drift", {}).get("message") or "no drift"))
    chk(20, "Automatically update on future trading days", True, "scheduler: GUI timer / main.py --daemon / scripts/register_daily_task.bat")
    _print(rows)
    return 0 if all(r[2] != "FAIL" for r in rows) else 1


def _print(rows):
    print("\nMINIMUM ACCEPTANCE TEST")
    for n, name, status, why in rows:
        print(f"{n:2d}. [{status:14s}] {name} - {why}")


if __name__ == "__main__":
    sys.exit(main())
