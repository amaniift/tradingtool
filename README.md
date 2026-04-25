# NSE EOD Analysis Dashboard

## Project Structure

```text
tradingtool/
├─ app.py                     # Streamlit frontend
├─ requirements.txt           # Python dependencies
├─ README.md
└─ src/
   ├─ __init__.py
   └─ data/
      ├─ __init__.py
      └─ data_manager.py      # EOD fetch + technicals + news sentiment
```

## Quick Start

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run the app:
   - `streamlit run app.py`

## Highlights

- Full NIFTY 50 stock universe in the selector.
- Fresh EOD retrieval with explicit start/end date download and refresh control.
- Multi-source latest data strategy: yfinance window + period + history, with NSE bhavcopy fallback and quote patching when needed.
- Data lag monitor in UI to detect stale closes.
- Multi-indicator technical stack:
   - RSI (14)
   - MACD (12, 26, 9)
   - SMA (20), SMA (50), EMA (20)
   - Daily return %
- News sentiment:
   - RSS from Economic Times + Moneycontrol
   - Ticker alias matching and fallback headlines
   - VADER sentiment scoring with pluggable scoring function
- Recommendation engine:
   - Weighted technical + sentiment score
   - BUY/SELL/HOLD thresholds
   - Confidence band + signal strength indicator
- NIFTY 50 Pulse tab:
   - Cross-sectional scan of all configured NIFTY 50 stocks
   - Top BUY/SELL candidates with score ranking
- Verdict tab:
   - Generates top 3 next-session trade setups (BUY/SELL)
   - Provides entry, target, and stop-loss for each setup
   - Supports Aggressive/Moderate/Conservative policy modes
   - Adds position sizing from account capital and risk-per-trade
   - Allows CSV download and verdict run audit logging
   - Uses score ranking plus ATR-based risk-reward logic
- Backtest tab:
   - Uses logged recommendation snapshots
   - Maps each signal to next trading day
   - Hit-rate, average return, cumulative return, and breakdown tables
- Derivatives + Quant Lab tab:
   - Pulls NSE NIFTY option-chain snapshot (nearest expiry)
   - Computes OI metrics: PCR, total CE/PE OI, max pain, gamma wall
   - Tracks ATM IV and rolling IV percentile from local history
   - Surfaces key strike magnets using OI + gamma concentration
   - Includes event-day risk templates (Normal / Expiry / Macro Event)
   - Builds cross-sectional feature store (technical + sentiment + lightweight fundamentals)
   - Produces probabilistic signal outputs and meta-model ranked recommendations
- Portfolio Lab tab:
   - Multi-position construction from verdict candidates
   - Correlation-aware capital allocation with max sector exposure control
   - Drawdown guardrails with VaR/CVaR risk estimates
   - Sector exposure and correlation matrix views
- Intraday Intelligence tab:
   - 5m/15m intraday timeframe support with configurable opening range window
   - Pre-open gap playbook and opening auction signal classification
   - Live session dashboard for pivot reactions and breakout failure statistics
- Modernized Streamlit UI with custom typography, gradient hero card, and improved information hierarchy.

## Current Logic

- EOD data uses `yfinance` with NSE tickers like `RELIANCE.NS`.
- Data reliability enhancements:
   - Attempts official NSE bhavcopy patch for latest finalized day
   - If latest historical close is missing, patches from quote fields as provisional data
   - Adds `DATA_SOURCE` and `IS_PROVISIONAL` columns in the returned EOD frame
- Technical indicators computed in `fetch_eod_data`:
  - RSI (14)
  - MACD (12, 26, 9)
   - SMA (20), SMA (50), EMA (20)
   - Daily return (%)
   - Uses `pandas-ta` when available, otherwise falls back to a pure-pandas implementation
- News sentiment in `fetch_news_sentiment`:
   - Attempts live RSS ingestion from Economic Times and Moneycontrol feeds
   - Filters headlines by expanded ticker aliases (company names, symbols, common variants)
   - Falls back to ticker-specific or generic mock headlines if no RSS matches are found
   - Scores headlines using VADER (`scorer` is injectable for easy LLM replacement)
- Recommendation logic in `app.py`:
   - Technical score uses RSI, MACD-vs-Signal, and trend-vs-SMA20
   - Combined score = `0.65 * technical_score + 0.35 * avg_news_sentiment`
   - BUY if score >= 0.25, SELL if score <= -0.25, else HOLD
   - Confidence band is derived from score magnitude and headline count
- Verdict engine logic in `app.py`:
   - Scores all NIFTY 50 stocks using RSI, MACD spread, trend regime, and 5-day momentum
   - Selects high-conviction BUY/SELL candidates by score threshold
   - Computes entry near current close and sets target/stop-loss using ATR-based risk bands
   - Applies policy-mode thresholds and ATR multipliers (Aggressive/Moderate/Conservative)
   - Calculates quantity using fixed risk budget per trade
   - Returns the top 3 setups by score strength and risk-reward quality
- Derivatives + advanced analytics logic:
   - Option-chain data from NSE API (`option-chain-indices`) filtered to nearest expiry
   - Max pain computed by minimizing expiry payoff pain across strikes
   - Gamma wall estimated from strike-level gamma exposure (`CE_gamma * CE_OI + PE_gamma * PE_OI`)
   - IV percentile computed from persisted ATM IV observations in `data/nifty_iv_history.csv`
   - Feature store persisted in `data/feature_store.csv` with:
      - Technical factors: RSI, MACD spread, SMA/EMA gaps, momentum, volatility, volume z-score
      - Sentiment factor: average ticker headline sentiment
      - Fundamental factors (when available): trailing PE, market cap, beta
   - Meta-model blend:
      - Rule score + ridge-regularized linear ML estimate of expected return
      - Probabilities for Up / Down / Flat and expected return estimate
- Portfolio construction logic:
   - Uses verdict pool as candidate universe
   - Penalizes highly-correlated names during allocation
   - Enforces sector exposure caps and allocation headroom
   - Computes portfolio VaR(95), CVaR(95), and max drawdown from historical daily returns
- Intraday logic:
   - Pulls intraday bars via yfinance (`5m`/`15m`)
   - Computes opening range high/low and breakout bias
   - Detects failed breakouts around R1/S1 pivot regions
   - Generates gap/auction playbook using session open vs previous close and early-bar momentum/volume

## Backtesting Log

- Click **Log Recommendation Snapshot** in the app to append current signal data.
- Logged file: `data/recommendation_log.csv`
- Fields include timestamp, ticker, recommendation, confidence, technical/sentiment scores, RSI, MACD, and news source.

## Verdict Log

- Click **Log Verdict Run** in the Verdict tab after generating setups.
- Logged file: `data/verdict_log.csv`
- Includes policy, capital, risk-per-trade, entry/target/stop, quantity, and projected PnL values.

## Next Upgrade Ideas

- Add more Indian finance/news RSS providers and source-level weighting.
- Replace VADER with an LLM API backend for context-aware sentiment.
- Add portfolio-level simulation (position sizing, costs, slippage).
