# VUSA AI Quant Terminal

An institutional-style quantitative **research and decision-support** application for the
Vanguard S&P 500 UCITS ETF (VUSA; `VUSAA` is accepted as an alias). It pulls in market, macro
and news data, validates it point-in-time, builds about 145 features, detects market regimes,
runs a multi-model probabilistic forecast stack with purged walk-forward validation, measures
risk, simulates scenarios and has 8 specialist agents (including a mandatory Skeptic) review
the evidence. The output is a transparent, two-dimensional **🟢 BUY / 🟡 HOLD / 🔴 SELL**
assessment: Opportunity and Risk scores, every component of the score, and the conditions that
would change it.

> **This is decision support, not a trading system and not investment advice.** It never places
> trades and never claims certainty. When data, models or sources are unavailable it says so.
> It does not fill gaps with invented values.

---

## Quick start (Windows, Python 3.10+)

```bat
install.bat          :: one time: creates .venv, installs requirements + PyTorch CPU, runs a self-test
run.bat              :: starts the GUI
```

In the GUI:
* **ANALYZE VUSA NOW** runs the 20-step daily pipeline in the selected mode (Fast / Balanced / Deep / Continuous).
* **RUN DEEP RESEARCH** runs every model including deep learning, nested Optuna tuning, all horizons, full
  validation (feature-causality audit), position-sizing research and the full Monte Carlo.
* **Reproduce analysis** re-runs a stored analysis with the same settings, seed and as-of date, and reports any
  differences.
* *Synthetic DEMO data* lets you explore the app without internet. It is labelled in red on every screen and in
  every report.

Headless use:

```bat
run.bat --cli                       :: one analysis, prints the daily report
run.bat --cli --mode deep --backtest
run.bat --cli --as-of 2024-06-28    :: historical analysis using only data up to that date
run.bat --daemon                    :: daily scheduler (trading days only)
scripts\register_daily_task.bat     :: Windows Task Scheduler job (weekdays 18:45)
python scripts\acceptance_test.py   :: the 20-point MINIMUM ACCEPTANCE TEST against live data
python -m pytest                    :: unit + integration test suite (offline, synthetic data)
```

Linux/macOS: `python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt torch && python main.py`.

## Configuration

Everything is configurable in the **Settings** tab and saved to `config/user_settings.json`. API keys
go only in `.env` (see `.env.example`):

| Key | Effect |
|---|---|
| `FRED_API_KEY` | **Recommended.** Enables true ALFRED vintages, so macro data carries real publication dates and revisions. Without it, the latest-vintage FRED CSV is used with conservative release lags, and the leakage audit reports the remaining revision risk as a `WARN`. |
| `ALPHAVANTAGE_API_KEY`, `TWELVEDATA_API_KEY` | Extra market-data fallbacks after yfinance → Stooq. |
| `NEWSAPI_KEY` | Adds NewsAPI to the RSS / official feeds (Fed, ECB, BLS, BEA, major financial media). |
| `ANTHROPIC_API_KEY` | Optional. Claude rephrases the explanation from the structured JSON. A fabrication guard checks every number in the text against the JSON and rejects the LLM output if any number cannot be traced. |

Fundamental valuation inputs (forward P/E, CAPE, EPS revisions) are not reliably available from free
APIs point-in-time. You enter them in *Settings → Valuation* with their date and source. Until you do,
the valuation component uses an explicitly-labelled **price-based proxy** at reduced weight.

Tax/cost assumptions (commission, spread, slippage, execution delay, tax on gains/dividends) are explicit,
editable and default to neutral values. The application makes no tax or legal claims (for example for Cyprus).

---

## Architecture (13 levels)

```
L1  Data ingestion        data/market, data/macro, data/news   providers + fallback chain + cache + source health
L2  Data validation       data/validation.py                   OHLC consistency, bad ticks, gaps vs exchange calendar, staleness
L3  Feature engineering   features/*                           ~145 causal features + lineage (source, transform, raw & z-value)
L4  Regime detection      regimes/                             8-regime rules, causal-filter HMM, GMM, change points, shocks
L5  Forecasting models    models/{statistical,tree,deep_learning}
L6  Probabilistic layer   models/calibration                   Platt/isotonic, conformal intervals, P(>5%), P(<-5%), P(DD>10%)
L7  Ensemble              models/ensemble                      best-single / equal / performance / stacked meta-model / BMA
L8  Risk engine           risk/, simulation/                   VaR/CVaR/Cornish-Fisher, stress, correlations, 6 Monte Carlo methods
L9  Signal engine         signals/                             12 components → Opportunity & Risk → hysteresis → confirmation
L10 Backtesting           backtesting/                         walk-forward retraining, next-open execution, costs, robustness
L11 Research agent        agents/                              8 specialist agents + Skeptic + decision aggregator
L12 Monitoring            monitoring/                          journal, live scorecard, model & feature drift, champion/challenger
L13 Explanation           explainability/, reports/            SHAP/permutation, stability, template/LLM text, daily report
```

The GUI (`gui/`) only talks to `services/api.py` (`Backend`). That service layer can later be exposed as a web API.
Storage is SQLite (`storage/vusa_quant.sqlite`) with portable SQL, so it can migrate to PostgreSQL. All 25 tables
from the specification exist, plus `feature_lineage` and `alerts`.

### Point-in-time guarantees

* Every daily bar is stamped `available_at` = official session close in UTC (timezone-normalised per exchange).
* Each decision time is the XETRA close plus a buffer. US closes (22:00 CET) are joined **as-of**, so a same-day US
  close is never visible to a European decision taken earlier that day.
* Macro values carry `effective_date`, `published_at`, `vintage` and `pit_method`. `daily_view()` returns only
  what had been published by each decision time. Revisions are tracked when vintages exist.
* News carries its publication timestamp. Items without a reliable timestamp are stamped with the retrieval
  time, so they can never move earlier.
* Labels use **executable** returns (entry at the next session's open), and carry `label_end` for purging.
* **Leakage audit** (runs before every forecast and backtest; any FAIL stops the backtest): future dates,
  future-price replication, target contamination, overlapping train/test labels, scaler leakage, feature
  causality (features recomputed on truncated data must not change), news-after-decision, macro-before-publication,
  and revision risk.

### Validation protocol

* Expanding purged walk-forward with an embargo of max(5, min(h, 20)) sessions; purged K-fold is available.
* Feature selection runs **inside each training fold**: coverage, MI, tree importance and (final fit) randomised-lasso
  stability, then a correlation filter. Selection frequency across folds is reported as Stable / Moderately stable / Unstable.
* Nested HPO (Deep mode): Optuna TPE runs on inner walk-forward splits of each outer training window only, with a
  robust objective (rank-IC + skill vs base rate − calibration error − instability − complexity).
* Ensemble methods are compared out-of-sample with their own expanding, non-overlapping weight estimation. The
  production method is chosen on the first two-thirds of that period.
* Calibration is fitted on early out-of-sample folds and evaluated on later ones. Conformal coverage is measured
  with a rolling, gap-respecting backtest.
* Every model is benchmarked against **naive drift** (Diebold-Mariano tests). ML confidence is skill **relative to
  that benchmark**, not relative to 50 %.

### Decision engine

Twelve independent dimensions are scored 0-100, each with evidence: trend, momentum, valuation, macro, sentiment,
volatility, breadth, regime, event risk, ML forecast, historical analogues, risk/reward. They give an **Opportunity**
score. The risk engine gives a separate **Risk** score. The final signal then applies:

* thresholds with **hysteresis** and a minimum duration, so the signal does not flip back and forth;
* **confirmation:** a STRONG signal needs agreement from ML, trend, risk/reward and regime. A lone bullish
  subsystem gives `HOLD / WEAK BULLISH`;
* **confidence penalties** for shocks, model drift, out-of-distribution inputs, stale data, synthetic data, audit
  failure and structural breaks. Below the minimum confidence, a BUY/SELL is shown as `HOLD (LOW CONFIDENCE - raw BUY)`;
* explicit **contradiction detection** ("Signal disagreement detected: …") and a **"What would change my mind?"** list
  (component levels, SMA200 price level, VIX, P(up) and downside thresholds).

---

## What was verified in the build environment (honest status)

The build sandbox had **no outbound access** to Yahoo, Stooq, FRED or news feeds. Everything below was verified with
the clearly-labelled synthetic generator, which uses the same code paths as live data:

* `python -m pytest` runs 50+ tests: calendar/holidays, point-in-time with revisions, US/EU close alignment, data
  validation, feature causality, the audit catching a centred moving average and an injected future return, purging,
  backtest **stopped** on leakage, next-open execution and costs, hysteresis, confirmation, low-confidence suppression,
  calibration, conformal coverage, Monte Carlo, the fabrication guard, the provider fallback chain, the no-data
  failure path (pipeline stops, nothing fabricated), the offline cache with a `DATA NOT CURRENT` flag, scheduler
  weekend/holiday skipping, the full 20-step pipeline, decision trace, journal resolution, drift, backtest +
  ablation, **bit-identical reproduction**, and a GUI smoke test.
* All model families trained and predicted inside the walk-forward loop: Elastic Net, RF, Extra Trees, HGB,
  XGBoost, LightGBM, CatBoost, ARIMA, SARIMA, structural state-space, ETS, LSTM, GRU, TCN, Transformer, N-BEATS,
  with nested Optuna.
* **Not yet verified against live providers.** Run `python scripts/acceptance_test.py` on a machine with internet;
  it reports each of the 20 acceptance criteria as PASS / FAIL / NOT VERIFIABLE.

On synthetic data the models are, as they should be, only slightly better than the naive benchmark, and sometimes
worse. The Skeptic and the self-audit report this. Expect the same humility on real data: the S&P 500 is close to
efficient at short horizons, and the system is designed to say so rather than hide it.

## Known limitations

* Daily data only. "Real-time" means the last completed session; delayed data is labelled.
* Breadth uses sector-ETF **proxies**, not constituent-level advance/decline data.
* Before VUSA's 2012 inception, history is extended with the S&P 500 converted to EUR. This is labelled as a proxy
  in charts and the self-audit, and excludes ETF fees and dividend timing.
* News sentiment has no free historical archive. It enters model training only after about 250 days of the app's own
  news memory; until then the ablation study reports "CANNOT TEST" for its value.
* The historical backtest decision rule uses ML + trend + momentum + volatility + regime only, because news and
  valuation cannot be reconstructed point-in-time.
* Without a FRED API key, macro history may contain later revisions (reported as `revision_risk: WARN`).
* A Temporal Fusion Transformer is not included (not practical for a single daily series). The causal Transformer
  encoder covers attention-based models. Deep-learning models are not used for 120D/252D horizons, where overlapping
  labels leave too few independent samples.

## Project layout

```
main.py  config/  core/  data/{market,macro,news,sources,point_in_time}  features/{technical,macro,sentiment,
cross_asset,breadth}  labeling/  models/{statistical,tree,deep_learning,ensemble,calibration}  regimes/  risk/
simulation/  backtesting/  validation/  explainability/  analysis/  signals/  agents/{quant,macro,news,technical,
risk,ml,skeptic,portfolio,decision}_agent  database/  alerts/  reports/  monitoring/  pipeline/  services/  gui/
tests/  notebooks/  scripts/  requirements.txt  .env.example  install.bat  run.bat
```
