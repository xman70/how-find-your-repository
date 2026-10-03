# VUSA AI DAILY INTELLIGENCE REPORT

> **SYNTHETIC DEMO DATA - NOT REAL MARKET DATA. Nothing in this report describes the real market.**

**Date (last session):** 2026-10-02  |  **Run:** DEEP_20261003_221329_48a322  |  **Mode:** deep

**Current price:** €558.07 (VUSA.DE)  |  **Data:** SYNTHETIC DEMO DATA - NOT REAL MARKET DATA

## Decision
| Signal | Opportunity | Risk | Confidence | Model agreement | Final score |
|---|---|---|---|---|---|
| **🟡 HOLD (LOW CONFIDENCE - raw BUY)** | 64/100 | 23/100 | 12% | 69% | 70 |

Quadrant: **HIGH OPPORTUNITY / LOW RISK**

Signal unchanged (1 session(s)).

## Forecasts (multi-horizon, probabilistic)
| Horizon | Expected return | P(up) | 80% interval | P(>+5%) | P(<-5%) | P(maxDD>10%) | Confidence |
|---|---|---|---|---|---|---|---|
| 1D | +0.1% | 53% | -1.2% .. +1.4% | 0% | 0% | 0% | 54% |
| 3D | +0.2% | 50% | -1.8% .. +2.8% | 2% | 1% | 0% | 53% |
| 5D | +0.3% | 56% | -2.4% .. +3.7% | 4% | 2% | 0% | 54% |
| 10D | +0.5% | 56% | -3.1% .. +5.5% | 13% | 5% | 2% | 49% |
| 20D | +1.0% | 59% | -4.4% .. +7.8% | 26% | 8% | 8% | 31% |
| 60D | +1.7% | 70% | -7.0% .. +14.9% | 43% | 14% | 27% | 33% |
| 120D | +5.4% | 68% | -5.0% .. +24.4% | 56% | 10% | 58% | 47% |
| 252D | +7.3% | 77% | -17.2% .. +44.1% | 74% | 16% | 83% | 30% |

**Forecast warnings:**
- 60D: Only ~12 independent 60D outcomes support the interval (overlapping labels): interval and probabilities are imprecise.
- 120D: Only ~12 independent 120D outcomes support the interval (overlapping labels): interval and probabilities are imprecise.
- 252D: Only ~12 independent 252D outcomes support the interval (overlapping labels): interval and probabilities are imprecise.

20D price range (80%): €533.35 → €601.56

## Model consensus (primary horizon)
- lstm: +2.20% (P(up) 53%), weight +0.06
- gru: +2.00% (P(up) 59%), weight +0.06
- transformer: +1.53% (P(up) 70%), weight +0.06
- tcn: +1.49% (P(up) 62%), weight +0.06
- sarima: +1.42% (P(up) 62%), weight +0.08
- state_space: +1.34% (P(up) 61%), weight +0.08
- extra_trees: +1.28% (P(up) 58%), weight +0.07
- arima: +1.15% (P(up) 59%), weight +0.08
- nbeats: +1.10% (P(up) 61%), weight +0.08
- naive_drift: +0.97% (P(up) 59%), weight +0.08
- random_forest: +0.77% (P(up) 55%), weight +0.06
- ets: +0.75% (P(up) 56%), weight +0.08
- catboost: -0.27% (P(up) 60%), weight +0.03
- xgboost: -0.40% (P(up) 52%), weight +0.05
- elastic_net: -0.76% (P(up) 53%), weight +0.01
- hist_gb: -0.94% (P(up) 56%), weight +0.03
- lightgbm: -1.49% (P(up) 60%), weight +0.04
- **ENSEMBLE (bayesian)**: +0.98%

## Market regime & volatility
Bull / MODERATE volatility (realised 20d 12.6%, percentile 35%); HMM stress probability 0%; drawdown -2.8%; 70 sessions in regime.

## Signal components (why the signal exists)
| Component | Score | Contribution | Evidence |
|---|---|---|---|
| trend | 66 | +1.9 | Price vs SMA200: +6.3%; SMA50 vs SMA200: +0.5% |
| momentum | 90 | +4.0 | 20d return +4.3%; 60d return +4.7% |
| valuation | 4 | -1.4 | PROXY: price vs 10-year log-trend +23.6% (not a fundamental valuation) |
| macro | 40 | -0.8 | Yield curve 10Y-3M: -2.42pp (inverted); High-yield OAS: 3.33% (20d chg -0.45) |
| sentiment | 61 | +0.6 | News sentiment unavailable or too few relevant articles; VIX z-score (1y): +0.1 (volatility sentiment) |
| volatility | 49 | -0.1 | Realised vol percentile (5y): 35%; 5d/60d vol ratio: 0.90 |
| breadth | 75 | +1.8 | Sectors above SMA200 (proxy): 82%; Sectors above SMA50 (proxy): 82% |
| regime | 78 | +2.8 | Rule-based regime: Bull (70 sessions); HMM stress-state probability: 0% |
| event_risk | 50 | +0.0 | 0 potentially bearish / 0 potentially bullish extracted events |
| ml_forecast | 70 | +2.5 | 20D expected return +0.98% (80% interval -4.4% .. +7.8%); Calibrated P(up) 59% (raw 59%; calibration: platt) |
| analogues | 34 | -1.2 | Top analogues: 20D weighted mean -0.9% vs unconditional +1.1%; Analogues positive after 20D: 40% (base rate 60%) |
| risk_reward | 86 | +3.6 | Upside P90 +7.8% vs downside P10 -4.4% (ratio 1.76); P(>+5%) 26% vs P(<-5%) 8% |

## Bull case
- [Risk Manager] Composite risk score 23/100
- [ML Scientist] 20D ensemble (bayesian) expected return +0.98%, calibrated P(up) 59%, 80% interval -4.4% .. +7.8%
- [Macro] High-yield OAS 3.33% (-0.45 over 20d)
- [Technical] Trend score 66/100: Price vs SMA200: +6.3%; SMA50 vs SMA200: +0.5%
- [Technical] Momentum score 90/100: 20d return +4.3%; 60d return +4.7%
- [Technical] Breadth score 75/100: Sectors above SMA200 (proxy): 82%; Sectors above SMA50 (proxy): 82%
- [Portfolio Analyst] Risk/reward score 86/100: Upside P90 +7.8% vs downside P10 -4.4% (ratio 1.76); P(>+5%) 26% vs P(<-5%) 8%
- [Skeptic] trend (66/100) argues bullish: Price vs SMA200: +6.3%; SMA50 vs SMA200: +0.5%

## Bear case
- [Macro] Sahm-style recession indicator 0.59 (trigger 0.50)
- [Quant] Nearest historical analogues: 20D weighted mean -0.9% (unconditional +1.1%), 40% positive
- [Skeptic] valuation (4/100) argues bearish: PROXY: price vs 10-year log-trend +23.6% (not a fundamental valuation)
- [Skeptic] macro (40/100) argues bearish: Yield curve 10Y-3M: -2.42pp (inverted); High-yield OAS: 3.33% (20d chg -0.45)
- [Skeptic] analogues (34/100) argues bearish: Top analogues: 20D weighted mean -0.9% vs unconditional +1.1%; Analogues positive after 20D: 40% (base rate 60%)
- [Skeptic] [Macro concern] Labour-market recession trigger is active.
- [Skeptic] [Risk Manager concern] Correlation breakdown vs VIX: +0.27 vs typical -0.08
- [Skeptic] [ML Scientist concern] Ensemble does NOT beat the naive drift benchmark out-of-sample (57.8% vs 58.6%).

## Neutral case
- [Technical] Volatility score 49/100: Realised vol percentile (5y): 35%; 5d/60d vol ratio: 0.90
- [Macro] CPI inflation 1.8% YoY, 6m trend +0.53%
- [Risk Manager] Current drawdown -2.8%; worst 1y drawdown -11.3%
- [Risk Manager] Monte Carlo (block bootstrap) 20D: P10 -5.5%, median +1.2%, P90 +7.9%; P(max drawdown > 10%) 7%
- [Quant] Up-day frequency last 60 sessions 48%
- [Risk Manager] 1-day 95% VaR 1.21% (historical), CVaR 1.45%; 20-day 95% VaR 7.9%
- [ML Scientist] Model sign agreement 69% across 16 models (spread of forecasts 1.08%)
- [Quant] Lag-1 daily return autocorrelation +0.022 (close to random walk)

## Main risks
- [Macro] Labour-market recession trigger is active.
- [Risk Manager] Correlation breakdown vs VIX: +0.27 vs typical -0.08
- [ML Scientist] Ensemble does NOT beat the naive drift benchmark out-of-sample (57.8% vs 58.6%).
- [Skeptic] The ensemble does not clearly beat the 'always expect the average drift' benchmark.
- [Skeptic] Probabilities are not well calibrated out-of-sample.
- [Skeptic] Data is NOT current - the assessment may be stale.
- [Skeptic] SYNTHETIC DEMO DATA - nothing here describes the real market.
- [Skeptic] Leakage audit WARN: revision_risk - 6 macro series use latest-vintage data with release-lag timing (CPIAUCSL, UNRATE, DGS10, DGS2, T10Y2Y, BAMLH0A0HYM2); later revisions may le

## Skeptic (counter-case)
- The ensemble does not clearly beat the 'always expect the average drift' benchmark.
- Probabilities are not well calibrated out-of-sample.
- Data is NOT current - the assessment may be stale.
- SYNTHETIC DEMO DATA - nothing here describes the real market.
- Leakage audit WARN: revision_risk - 6 macro series use latest-vintage data with release-lag timing (CPIAUCSL, UNRATE, DGS10, DGS2, T10Y2Y, BAMLH0A0HYM2); later revisions may le
- Signal disagreement detected: bullish = trend, momentum, sentiment, breadth, regime, ml_forecast, risk_reward; bearish = valuation, macro, analogues

## ⚠ Contradictions
- Signal disagreement detected: bullish = trend, momentum, sentiment, breadth, regime, ml_forecast, risk_reward; bearish = valuation, macro, analogues

## What would change the signal?
- Final score 70: needs +-8 points for BUY or -32 for SELL.
- Trend break level: price below SMA200 at 525.18 (-5.9% from 558.07).
- Volatility threshold: VIX above 30 would cut the volatility component sharply.
- ML threshold: calibrated 20D P(up) falling below 50% (now 59%).
- Downside threshold: P(20D return < -5%) above 25% (now 8%).
- A regime change (e.g. to Bear / High volatility), model-drift alarm, or a sharp rise in model disagreement would lower confidence and may change the signal.

## Macro condition
- cpi_yoy: 0.018
- unrate: 3.722
- hy_oas: 3.328

## News sentiment & major events
News unavailable: synthetic demo mode - no real news used

## Historical analogues
| Date | Similarity | Regime | 5D | 20D | 60D | Max DD 60D |
|---|---|---|---|---|---|---|
| 2006-09-27 | 0.65 | Bull | -0.1% | -9.5% | +8.5% | -11.6% |
| 2015-09-30 | 0.65 | Bull | +4.2% | +10.2% | +9.6% | -9.3% |
| 2016-01-26 | 0.60 | Bull | -3.5% | +2.0% | +2.3% | -3.5% |
| 2015-08-21 | 0.61 | Strong bull | -1.1% | -0.8% | +16.3% | -4.2% |
| 2007-02-06 | 0.60 | Strong bull | -0.4% | -4.0% | +0.8% | -4.9% |
| 2007-04-23 | 0.61 | Strong bull | +1.4% | +3.5% | +3.4% | -4.1% |
| 2014-03-20 | 0.59 | Bull | +2.4% | -3.8% | -11.4% | -13.4% |
| 2017-12-08 | 0.59 | Strong bull | +2.2% | +3.1% | +8.7% | -4.9% |

## Stress scenarios (SCENARIOS, not predictions)
- Normal (median month): +1.1% (historical median 20d return)
- Mild correction: -4.3% (beta to S&P (20d) = 0.86)
- Bear market: -17.1% (beta to S&P (20d) = 0.86)
- Crash: -28.3% (beta to S&P (20d) = 0.86)
- Rate shock: +13.9% (rate sensitivity +0.139/pp)
- Inflation shock: +10.4% (rate sensitivity +0.139/pp; VIX sensitivity +0.0000/pt)
- Volatility shock: +0.0% (VIX sensitivity +0.0000/pt)
- USD rally (EUR investor): -1.6% (EUR/USD sensitivity +0.20)
- USD slump (EUR investor): +1.6% (EUR/USD sensitivity +0.20)

## Model performance (out-of-sample, primary horizon)
- best_single: DA 52.1%, IC 0.022, RMSE 0.0587, Brier 0.2486 (n=4340)
- equal_weight: DA 57.0%, IC -0.011, RMSE 0.0602, Brier 0.2502 (n=4340)
- performance_weighted: DA 57.3%, IC -0.009, RMSE 0.0595, Brier 0.2477 (n=4340)
- stacked: DA 52.0%, IC -0.011, RMSE 0.0598, Brier 0.2668 (n=4340)
- bayesian: DA 57.8%, IC -0.006, RMSE 0.0589, Brier 0.2457 (n=4340)
- Calibration (platt): ECE raw 0.038 → 0.072; conformal coverage 79%

## Data quality
Score 100/100. Leakage audit: PASSED.
- SYNTHETIC DEMO DATA: generated locally, NOT real market data. Use only to explore the application.

## Self-audit

**MODEL STRENGTHS**
- Conformal coverage 79% vs nominal 80%.
- Best base model by IC: ets (+0.056).

**MODEL WEAKNESSES**
- Ensemble OOS directional accuracy 57.8% vs naive drift 58.6% (20D, n=4340).
- Ensemble OOS rank IC -0.006.
- OOS calibration ECE 0.072 (method platt).
- Weakest base model by IC: naive_drift (-0.125).

**KNOWN LIMITATIONS**
- Daily data only; intraday dynamics and overnight US moves after the XETRA close are not modelled at decision time.
- Breadth uses sector-ETF proxies, not constituent-level data.
- Forecasts are probabilistic and regime-dependent; past relationships may break.
- Historical backtest decision rule excludes news/valuation (not reconstructable point-in-time).

**DATA LIMITATIONS**
- SYNTHETIC DEMO DATA - results say nothing about the real market.
- No current news: synthetic demo mode - no real news used
- No fundamental valuation inputs (P/E, CAPE) - valuation uses a price-based proxy.

**POSSIBLE LEAKAGE**
- [WARN] revision_risk: 6 macro series use latest-vintage data with release-lag timing (CPIAUCSL, UNRATE, DGS10, DGS2, T10Y2Y, BAMLH0A0HYM2); later revisions may leak. Add a FRED API key for true ALFRED vintages.

**REGIME DEPENDENCE**
- naive_drift accuracy by regime: Bear 50% (n=129), Neutral 66% (n=421), Strong bear 61% (n=798), Recovery 36% (n=107), Bull 59% (n=1910), High volatility 44% (n=402), Crisis 60% (n=216), Strong bull 64% (n=607)

**MODEL DISAGREEMENT**
- Sign agreement 69%, forecast spread 1.08%

**RECENT PERFORMANCE**
- 0 live scorecard rows

**OUT-OF-SAMPLE PERFORMANCE**
- best_single: DA 52.1%, IC +0.022
- equal_weight: DA 57.0%, IC -0.011
- performance_weighted: DA 57.3%, IC -0.009
- stacked: DA 52.0%, IC -0.011
- bayesian: DA 57.8%, IC -0.006

**RESEARCH RISKS**
- Multiple models / ensembles / thresholds were compared: some selection bias remains even with nested selection; treat the OOS edge as an upper bound.
- Overlapping multi-day labels make effective sample sizes much smaller than row counts.

## Explanation
WHAT: 🟡 HOLD (LOW CONFIDENCE - raw BUY) - opportunity 63.6/100, risk 23.2/100, confidence 12%.
The 20D ensemble expects +1.0% with an 80% interval of -4.4% to +7.8%; calibrated P(up) 59%.
WHY: momentum +4.0 pts; risk_reward +3.6 pts; regime +2.8 pts; ml_forecast +2.5 pts; trend +1.9 pts; breadth +1.8 pts.
HOW CONFIDENT: component agreement 58%; penalties: out_of_distribution 3%, data_not_current 30%, synthetic_data 50%.
WHAT COULD INVALIDATE IT: Final score 70: needs +-8 points for BUY or -32 for SELL. | Trend break level: price below SMA200 at 525.18 (-5.9% from 558.07). | Volatility threshold: VIX above 30 would cut the volatility component sharply.
HISTORICALLY: closest analogues averaged -0.9% over 20D (40% positive).
MODELS: 69% of 16 models agree on direction.
NOTE: Signal disagreement detected: bullish = trend, momentum, sentiment, breadth, regime, ml_forecast, risk_reward; bearish = valuation, macro, analogues
This is a probabilistic research assessment, not a guarantee or investment advice.

---
*Decision-support research output. Probabilistic, not a guarantee and not investment advice. No trades are placed by this application.*