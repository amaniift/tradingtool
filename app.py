"""Streamlit dashboard for NSE EOD and news-sentiment analysis."""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data.data_manager import fetch_eod_data, fetch_news_sentiment, get_yfinance_session


NIFTY_50_STOCKS = {
    "Adani Enterprises": "ADANIENT.NS",
    "Adani Ports & SEZ": "ADANIPORTS.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Axis Bank": "AXISBANK.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Bharat Electronics": "BEL.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Cipla": "CIPLA.NS",
    "Coal India": "COALINDIA.NS",
    "Dr. Reddy's": "DRREDDY.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Eternal": "ETERNAL.NS",
    "Grasim Industries": "GRASIM.NS",
    "HCLTech": "HCLTECH.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "HDFC Life": "HDFCLIFE.NS",
    "Hindalco": "HINDALCO.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "IndiGo": "INDIGO.NS",
    "Infosys": "INFY.NS",
    "ITC": "ITC.NS",
    "Jio Financial Services": "JIOFIN.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Larsen & Toubro": "LT.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Max Healthcare": "MAXHEALTH.NS",
    "Nestle India": "NESTLEIND.NS",
    "NTPC": "NTPC.NS",
    "ONGC": "ONGC.NS",
    "Power Grid": "POWERGRID.NS",
    "Reliance Industries": "RELIANCE.NS",
    "SBI Life": "SBILIFE.NS",
    "Shriram Finance": "SHRIRAMFIN.NS",
    "State Bank of India": "SBIN.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Tata Consumer Products": "TATACONSUM.NS",
    "Tata Motors Passenger Vehicles": "TMPV.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Tech Mahindra": "TECHM.NS",
    "Titan": "TITAN.NS",
    "Trent": "TRENT.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Wipro": "WIPRO.NS",
}

LOG_PATH = Path("data") / "recommendation_log.csv"
VERDICT_LOG_PATH = Path("data") / "verdict_log.csv"
HOLD_RETURN_BAND = 0.005

VERDICT_POLICIES = {
    "Aggressive": {
        "entry_buffer": 0.0005,
        "score_threshold": 0.16,
        "stop_atr": 1.00,
        "target_atr": 2.20,
    },
    "Moderate": {
        "entry_buffer": 0.0010,
        "score_threshold": 0.22,
        "stop_atr": 1.10,
        "target_atr": 1.80,
    },
    "Conservative": {
        "entry_buffer": 0.0015,
        "score_threshold": 0.30,
        "stop_atr": 1.25,
        "target_atr": 1.55,
    },
}


def inject_modern_theme() -> None:
    """Apply a premium dark, highly legible theme to the Streamlit app tailored for trading tools."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

        :root {
            --bg-soft: #0b1120;
            --bg-card: #161e31;
            --ink-strong: #f8fafc;
            --ink-muted: #94a3b8;
            --ink-soft: #64748b;
            --accent: #3b82f6;
            --accent-2: #10b981;
            --line: #334155;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4), 0 2px 4px -1px rgba(0, 0, 0, 0.24);
        }

        .stApp {
            background-color: var(--bg-soft);
            background-image: 
                radial-gradient(circle at 100% 0%, rgba(59, 130, 246, 0.1), transparent 30%),
                radial-gradient(circle at 0% 100%, rgba(16, 185, 129, 0.08), transparent 30%);
            color: var(--ink-strong);
            font-family: 'Inter', sans-serif;
        }

        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3, h4 {
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: -0.02em;
            color: var(--ink-strong);
            margin-top: 0.5rem;
        }

        p, label, li, span, small, [data-testid="stMarkdownContainer"] {
            color: var(--ink-muted);
        }

        .hero-card {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 2rem;
            color: var(--ink-strong);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.2);
            margin-bottom: 2rem;
        }

        .hero-title {
            font-size: 1.5rem;
            font-weight: 800;
            margin: 0;
            background: linear-gradient(135deg, #60a5fa 0%, #34d399 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-sub {
            margin: 0.5rem 0 0;
            color: var(--ink-muted);
            font-size: 0.95rem;
        }

        section[data-testid="stSidebar"] {
            background-color: #0f172a;
            border-right: 1px solid #1e293b;
        }

        section[data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }

        section[data-testid="stSidebar"] [data-baseweb="select"] > div,
        section[data-testid="stSidebar"] [data-baseweb="base-input"] > div,
        section[data-testid="stSidebar"] .stButton > button {
            background-color: #1e293b;
            border: 1px solid #334155;
            box-shadow: none;
            border-radius: 8px;
        }

        section[data-testid="stSidebar"] .stButton > button:hover {
            border-color: #3b82f6;
            background-color: #0f172a;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            border-bottom: 2px solid #1e293b;
        }

        .stTabs [data-baseweb="tab"] {
            color: #94a3b8 !important;
            background: transparent;
            border: none;
            padding: 0.75rem 1rem;
            font-weight: 600;
            font-family: 'Space Grotesk', sans-serif;
            transition: all 0.2s ease;
        }

        .stTabs [aria-selected="true"] {
            color: #f8fafc !important;
            background: transparent;
            border-bottom: 2px solid #3b82f6 !important;
            box-shadow: none;
        }

        [data-testid="stMetric"] {
            background: rgba(30, 41, 59, 0.5);
            backdrop-filter: blur(12px);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1rem 1.25rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
            border-color: #475569;
        }

        [data-testid="stMetricLabel"] {
            color: #94a3b8;
            font-weight: 500;
            font-size: 0.875rem;
            letter-spacing: 0.02em;
        }

        [data-testid="stMetricValue"] {
            color: #f8fafc;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 1.5rem;
        }

        /* Enhanced scrollbar for premium feel */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0f172a;
        }
        ::-webkit-scrollbar-thumb {
            background: #334155;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #475569;
        }

        .stButton > button {
            border-radius: 8px;
            border: 1px solid #334155;
            background: #1e293b;
            color: #f8fafc;
            font-weight: 600;
            padding: 0.5rem 1rem;
            transition: all 0.2s ease;
        }

        .stButton > button:hover {
            border-color: #3b82f6;
            background: #0f172a;
            color: #60a5fa;
            transform: translateY(-1px);
        }

        [data-baseweb="base-input"] > div,
        [data-baseweb="select"] > div,
        .stNumberInput > div > div {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            color: #f8fafc;
        }

        .stSlider [data-baseweb="slider"] div[role="slider"] {
            border-color: #3b82f6;
            background: #3b82f6;
        }
        
        .stSlider [data-baseweb="slider"] div[data-testid="stTickBar"] {
            background-color: #334155;
        }

        div[data-testid="stExpander"] {
            background: #161e31;
            border: 1px solid #334155;
            border-radius: 12px;
        }

        div[data-testid="stExpander"] summary {
            color: #f8fafc;
            font-weight: 600;
        }
        
        div[data-testid="stExpander"] summary:hover {
            color: #60a5fa;
        }
        
        .stAlert {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            color: #f8fafc;
        }
                }

                .stAlert {
                    border-radius: 12px;
                    border: 1px solid #d6e1ee;
                    color: var(--ink-strong);
                }

                .stDataFrame, div[data-testid="stTable"] {
                    border: 1px solid #d2ddec;
                    border-radius: 12px;
                    overflow: hidden;
                }

                .stDataFrame * {
                    color: #1b304b !important;
                }

                .stCaption {
                    color: var(--ink-soft) !important;
                    font-size: 0.9rem;
                }

                @media (max-width: 900px) {
                    .block-container {
                        padding-top: 0.8rem;
                    }

                    .hero-card {
                        padding: 1rem;
                    }

                    .hero-title {
                        font-size: 1rem;
                    }

                    .stTabs [data-baseweb="tab"] {
                        font-size: 0.86rem;
                        padding: 0.46rem 0.64rem;
                    }
                }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _safe_float(value: float | None, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return default
    if value_f != value_f:
        return default
    return value_f


@st.cache_data(ttl=600, show_spinner=False)
def get_eod_cached(ticker: str, days: int) -> pd.DataFrame:
    return fetch_eod_data(ticker, days=days)


@st.cache_data(ttl=900, show_spinner=False)
def get_news_cached(ticker: str, max_items: int, use_live_news: bool) -> pd.DataFrame:
    return fetch_news_sentiment(ticker, max_items=max_items, use_live_news=use_live_news)


def generate_recommendation(
    rsi_value: float,
    macd_value: float,
    macd_signal_value: float,
    close_value: float,
    sma_20_value: float,
    avg_news_sentiment: float,
) -> tuple[str, float, float, float]:
    """Blend technical and sentiment signals into BUY/SELL/HOLD."""
    rsi_signal = 1.0 if rsi_value < 30 else -1.0 if rsi_value > 70 else 0.0
    macd_signal = 1.0 if macd_value > macd_signal_value else - \
        1.0 if macd_value < macd_signal_value else 0.0
    trend_signal = 1.0 if close_value > sma_20_value else -1.0

    technical_score = (0.45 * rsi_signal) + \
        (0.35 * macd_signal) + (0.20 * trend_signal)
    combined_score = (0.65 * technical_score) + (0.35 * avg_news_sentiment)

    if combined_score >= 0.25:
        return "BUY", combined_score, technical_score, avg_news_sentiment
    if combined_score <= -0.25:
        return "SELL", combined_score, technical_score, avg_news_sentiment
    return "HOLD", combined_score, technical_score, avg_news_sentiment


def derive_confidence_band(combined_score: float, headline_count: int) -> str:
    """Estimate recommendation confidence from score magnitude and headline depth."""
    abs_score = abs(combined_score)
    if headline_count >= 6 and abs_score >= 0.55:
        return "High"
    if headline_count >= 4 and abs_score >= 0.30:
        return "Medium"
    return "Low"


def log_recommendation_snapshot(
    ticker: str,
    company_name: str,
    recommendation: str,
    confidence_band: str,
    combined_score: float,
    technical_score: float,
    sentiment_score: float,
    rsi_value: float,
    macd_value: float,
    macd_signal_value: float,
    headline_count: int,
    news_source: str,
) -> Path:
    """Append recommendation snapshot to CSV for historical analysis/backtesting."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ticker": ticker,
        "company_name": company_name,
        "recommendation": recommendation,
        "confidence": confidence_band,
        "combined_score": round(combined_score, 4),
        "technical_score": round(technical_score, 4),
        "sentiment_score": round(sentiment_score, 4),
        "rsi_14": round(rsi_value, 4),
        "macd": round(macd_value, 4),
        "macd_signal": round(macd_signal_value, 4),
        "headline_count": headline_count,
        "news_source": news_source,
    }

    fieldnames = list(row.keys())
    write_header = not LOG_PATH.exists()

    with LOG_PATH.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return LOG_PATH


def load_recommendation_log() -> pd.DataFrame:
    """Load historical recommendation snapshots if available."""
    if not LOG_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(LOG_PATH)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_price_history_for_backtest(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """Fetch close-price history for backtest outcome mapping."""
    import yfinance as yf

    price_df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False,
        session=get_yfinance_session(),
    )
    if price_df.empty:
        return pd.Series(dtype="float64")

    if isinstance(price_df.columns, pd.MultiIndex):
        price_df.columns = price_df.columns.get_level_values(0)

    close = price_df["Close"].dropna().copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def _strategy_return(recommendation: str, realized_return: float) -> float:
    if recommendation == "BUY":
        return realized_return
    if recommendation == "SELL":
        return -realized_return
    return 0.0


def _is_hit(recommendation: str, realized_return: float) -> bool:
    if recommendation == "BUY":
        return realized_return > 0
    if recommendation == "SELL":
        return realized_return < 0
    return abs(realized_return) <= HOLD_RETURN_BAND


def build_backtest_results(log_df: pd.DataFrame) -> pd.DataFrame:
    """Map each logged recommendation to the next trading day's outcome."""
    if log_df.empty:
        return pd.DataFrame()

    required = {"timestamp", "ticker", "recommendation",
                "confidence", "combined_score"}
    if not required.issubset(set(log_df.columns)):
        return pd.DataFrame()

    work_df = log_df.copy()
    work_df["signal_datetime"] = pd.to_datetime(
        work_df["timestamp"], errors="coerce")
    work_df = work_df.dropna(
        subset=["signal_datetime", "ticker", "recommendation"]).copy()
    if work_df.empty:
        return pd.DataFrame()

    work_df["signal_date"] = work_df["signal_datetime"].dt.floor("D")
    start_dt = (work_df["signal_date"].min() -
                pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    end_dt = (work_df["signal_date"].max() +
              pd.Timedelta(days=15)).strftime("%Y-%m-%d")

    close_by_ticker: dict[str, pd.Series] = {}
    for ticker in work_df["ticker"].dropna().unique():
        close_by_ticker[str(ticker)] = fetch_price_history_for_backtest(
            ticker, start_dt, end_dt)

    results = []
    for row in work_df.itertuples(index=False):
        ticker = str(row.ticker)
        close = close_by_ticker.get(ticker, pd.Series(dtype="float64"))
        if close.empty:
            continue

        signal_date = pd.Timestamp(row.signal_date)
        if signal_date.tzinfo is not None:
            signal_date = signal_date.tz_localize(None)

        entry_idx = close.index.searchsorted(signal_date)
        if entry_idx >= len(close) or entry_idx + 1 >= len(close):
            continue

        entry_date = close.index[entry_idx]
        next_date = close.index[entry_idx + 1]
        entry_price = float(close.iloc[entry_idx])
        next_price = float(close.iloc[entry_idx + 1])
        realized_return = (next_price - entry_price) / entry_price
        recommendation = str(row.recommendation).upper()
        strat_return = _strategy_return(recommendation, realized_return)
        is_hit = _is_hit(recommendation, realized_return)

        results.append(
            {
                "timestamp": row.timestamp,
                "ticker": ticker,
                "recommendation": recommendation,
                "confidence": row.confidence,
                "combined_score": float(row.combined_score),
                "entry_date": entry_date.date().isoformat(),
                "next_date": next_date.date().isoformat(),
                "entry_close": round(entry_price, 2),
                "next_close": round(next_price, 2),
                "next_day_return": realized_return,
                "strategy_return": strat_return,
                "is_hit": is_hit,
            }
        )

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)


@st.cache_data(ttl=1800, show_spinner=False)
def build_nifty50_pulse(days: int, finalized_only: bool) -> pd.DataFrame:
    """Create a technical pulse table for all configured NIFTY 50 stocks."""
    rows = []
    for company, ticker in NIFTY_50_STOCKS.items():
        eod_df = fetch_eod_data(ticker, days=max(days, 55))
        if eod_df.empty:
            continue

        if finalized_only and "IS_PROVISIONAL" in eod_df.columns:
            finalized_df = eod_df[~eod_df["IS_PROVISIONAL"]].copy()
            if not finalized_df.empty:
                eod_df = finalized_df

        latest = eod_df.iloc[-1]
        if len(eod_df) < 6:
            continue
        prev_close = _safe_float(eod_df.iloc[-2].get("Close"), 0.0)
        close_value = _safe_float(latest.get("Close"), 0.0)
        if prev_close <= 0 or close_value <= 0:
            continue

        day_change = ((close_value - prev_close) / prev_close) * 100
        five_day_base = _safe_float(eod_df.iloc[-6].get("Close"), close_value)
        five_day_change = ((close_value - five_day_base) /
                           five_day_base) * 100 if five_day_base > 0 else 0.0
        rsi_value = _safe_float(latest.get("RSI_14"), 50.0)
        macd_value = _safe_float(latest.get("MACD"), 0.0)
        macd_signal_value = _safe_float(latest.get("MACD_SIGNAL"), 0.0)
        sma_20_value = _safe_float(latest.get("SMA_20"), close_value)
        data_source = str(latest.get("DATA_SOURCE", "unknown"))
        is_provisional = bool(latest.get("IS_PROVISIONAL", False))

        recommendation, combined_score, technical_score, _ = generate_recommendation(
            rsi_value=rsi_value,
            macd_value=macd_value,
            macd_signal_value=macd_signal_value,
            close_value=close_value,
            sma_20_value=sma_20_value,
            avg_news_sentiment=0.0,
        )

        rows.append(
            {
                "company": company,
                "ticker": ticker,
                "close": round(close_value, 2),
                "day_change_pct": round(day_change, 2),
                "five_day_change_pct": round(five_day_change, 2),
                "rsi_14": round(rsi_value, 2),
                "combined_score": round(combined_score, 3),
                "technical_score": round(technical_score, 3),
                "recommendation": recommendation,
                "data_source": data_source,
                "bar_status": "Provisional" if is_provisional else "Final",
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("combined_score", ascending=False)


def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Compute ATR-like volatility proxy from OHLC data."""
    if df.empty or len(df) < period + 2:
        return 0.0

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)

    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(period).mean().iloc[-1]
    if pd.isna(atr):
        return 0.0
    return float(atr)


def _cluster_price_points(
    points: list[tuple[pd.Timestamp, float]],
    tolerance: float,
) -> list[dict[str, object]]:
    """Cluster nearby swing points into stable price bands."""
    if not points:
        return []

    clusters: list[dict[str, object]] = []
    for point_date, point_price in sorted(points, key=lambda item: item[1]):
        placed = False
        for cluster in clusters:
            if abs(point_price - float(cluster["level"])) <= tolerance:
                cluster_points = cluster["points"]
                cluster_points.append((point_date, point_price))
                level_prices = [price for _, price in cluster_points]
                cluster["level"] = sum(level_prices) / len(level_prices)
                cluster["points"] = cluster_points
                placed = True
                break

        if not placed:
            clusters.append(
                {
                    "level": point_price,
                    "points": [(point_date, point_price)],
                }
            )

    return clusters


def _calculate_support_resistance_levels(
    eod_df: pd.DataFrame,
    current_close: float,
    lookback_days: int = 120,
    pivot_window: int = 3,
    max_levels: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate support and resistance using clustered swing pivots.

    The calculation uses recent swing highs/lows, ATR-aware clustering,
    and touch/recency scoring so the output focuses on levels that have
    been respected more than once instead of isolated extremes.
    """
    required_cols = {"High", "Low", "Close"}
    if eod_df.empty or not required_cols.issubset(set(eod_df.columns)):
        return pd.DataFrame(), pd.DataFrame()

    recent_df = eod_df.tail(min(len(eod_df), lookback_days)).copy()
    if len(recent_df) < (pivot_window * 2) + 1 or current_close <= 0:
        return pd.DataFrame(), pd.DataFrame()

    recent_high = recent_df["High"].astype(float)
    recent_low = recent_df["Low"].astype(float)
    atr = _compute_atr(recent_df, period=min(14, max(5, len(recent_df) // 4)))
    tolerance = max(current_close * 0.006, atr * 0.6 if atr > 0 else current_close * 0.006)
    window_size = (pivot_window * 2) + 1

    low_roll = recent_low.rolling(window=window_size, center=True).min()
    high_roll = recent_high.rolling(window=window_size, center=True).max()

    support_points: list[tuple[pd.Timestamp, float]] = []
    resistance_points: list[tuple[pd.Timestamp, float]] = []

    for idx, low_price in recent_low.items():
        roll_value = low_roll.loc[idx]
        if pd.notna(roll_value) and float(low_price) <= float(roll_value) + 1e-9:
            support_points.append((pd.Timestamp(idx), float(low_price)))

    for idx, high_price in recent_high.items():
        roll_value = high_roll.loc[idx]
        if pd.notna(roll_value) and float(high_price) >= float(roll_value) - 1e-9:
            resistance_points.append((pd.Timestamp(idx), float(high_price)))

    def _build_levels(
        points: list[tuple[pd.Timestamp, float]],
        side: str,
    ) -> pd.DataFrame:
        clusters = _cluster_price_points(points, tolerance)
        level_rows: list[dict[str, object]] = []

        for cluster in clusters:
            cluster_points = cluster["points"]
            cluster_level = float(cluster["level"])
            if side == "support" and cluster_level >= current_close:
                continue
            if side == "resistance" and cluster_level <= current_close:
                continue

            weighted_prices = []
            weighted_scores = []
            for point_date, point_price in cluster_points:
                age_days = max((recent_df.index[-1] - pd.Timestamp(point_date)).days, 0)
                recency_weight = 1.0 + max(0.0, (lookback_days - age_days) / lookback_days)
                weighted_prices.append(point_price * recency_weight)
                weighted_scores.append(recency_weight)

            cluster_level = sum(weighted_prices) / sum(weighted_scores)
            if side == "support":
                side_series = recent_low
                distance_pct = ((current_close - cluster_level) / current_close) * 100.0
            else:
                side_series = recent_high
                distance_pct = ((cluster_level - current_close) / current_close) * 100.0

            touch_mask = (side_series.sub(cluster_level).abs() <= tolerance)
            touch_count = int(touch_mask.sum())
            if touch_count == 0:
                continue

            last_actual_touch = recent_df.index[touch_mask].max()
            last_touch_days = int((recent_df.index[-1] - last_actual_touch).days)
            recency_bonus = max(0.0, 1.0 - (last_touch_days / max(lookback_days, 1)))
            strength = (touch_count * (1.0 + recency_bonus)) + (len(cluster_points) * 0.35)

            level_rows.append(
                {
                    "Level Type": side.title(),
                    "Level": round(cluster_level, 2),
                    "Distance to Close (%)": round(distance_pct, 2),
                    "Touches": touch_count,
                    "Last Touch": pd.Timestamp(last_actual_touch).date().isoformat(),
                    "Strength": round(strength, 2),
                }
            )

        if not level_rows:
            return pd.DataFrame()

        levels_df = pd.DataFrame(level_rows)
        if side == "support":
            levels_df = levels_df.sort_values("Level", ascending=False)
        else:
            levels_df = levels_df.sort_values("Level", ascending=True)
        return levels_df.head(max_levels).reset_index(drop=True)

    return _build_levels(support_points, "support"), _build_levels(resistance_points, "resistance")


@st.cache_data(ttl=1800, show_spinner=False)
def build_verdict_candidates(
    days: int,
    finalized_only: bool,
    policy_name: str,
) -> pd.DataFrame:
    """Generate ranked next-session trade candidates with risk-managed levels."""
    candidates = []
    policy = VERDICT_POLICIES.get(policy_name, VERDICT_POLICIES["Moderate"])
    entry_buffer = float(policy["entry_buffer"])
    score_threshold = float(policy["score_threshold"])
    stop_atr = float(policy["stop_atr"])
    target_atr = float(policy["target_atr"])

    for company, ticker in NIFTY_50_STOCKS.items():
        eod_df = fetch_eod_data(ticker, days=max(days, 90))
        if eod_df.empty:
            continue

        if finalized_only and "IS_PROVISIONAL" in eod_df.columns:
            finalized_df = eod_df[~eod_df["IS_PROVISIONAL"]].copy()
            if not finalized_df.empty:
                eod_df = finalized_df

        if len(eod_df) < 25:
            continue

        latest = eod_df.iloc[-1]
        close_value = _safe_float(latest.get("Close"), 0.0)
        sma_20_value = _safe_float(latest.get("SMA_20"), close_value)
        sma_50_value = _safe_float(latest.get("SMA_50"), close_value)
        rsi_value = _safe_float(latest.get("RSI_14"), 50.0)
        macd_value = _safe_float(latest.get("MACD"), 0.0)
        macd_signal_value = _safe_float(latest.get("MACD_SIGNAL"), 0.0)

        if close_value <= 0:
            continue

        momentum_5 = float(eod_df["Close"].pct_change(
            5).iloc[-1]) if len(eod_df) > 6 else 0.0
        atr = _compute_atr(eod_df, period=14)
        if atr <= 0:
            atr = close_value * 0.008

        # Score model: trend + momentum + oscillator + MACD spread.
        rsi_component = max(min((rsi_value - 50.0) / 20.0, 1.0), -1.0)
        macd_component = 1.0 if macd_value > macd_signal_value else -1.0
        trend20_component = 1.0 if close_value >= sma_20_value else -1.0
        trend50_component = 1.0 if sma_20_value >= sma_50_value else -1.0
        momentum_component = max(min(momentum_5 / 0.04, 1.0), -1.0)

        score = (
            0.28 * rsi_component
            + 0.24 * macd_component
            + 0.20 * trend20_component
            + 0.13 * trend50_component
            + 0.15 * momentum_component
        )

        if score >= score_threshold:
            side = "BUY"
            entry = close_value * (1.0 + entry_buffer)
            stop_loss = entry - (stop_atr * atr)
            target = entry + (target_atr * atr)
        elif score <= -score_threshold:
            side = "SELL"
            entry = close_value * (1.0 - entry_buffer)
            stop_loss = entry + (stop_atr * atr)
            target = entry - (target_atr * atr)
        else:
            continue

        reward = abs(target - entry)
        risk = abs(entry - stop_loss)
        rr_ratio = reward / risk if risk > 0 else 0.0

        candidates.append(
            {
                "company": company,
                "ticker": ticker,
                "side": side,
                "score": round(score, 3),
                "close": round(close_value, 2),
                "entry": round(entry, 2),
                "target": round(target, 2),
                "stop_loss": round(stop_loss, 2),
                "rr_ratio": round(rr_ratio, 2),
                "rsi_14": round(rsi_value, 2),
                "macd_spread": round(macd_value - macd_signal_value, 3),
                "atr_14": round(atr, 2),
                "data_source": str(latest.get("DATA_SOURCE", "yfinance")),
                "bar_status": "Provisional" if bool(latest.get("IS_PROVISIONAL", False)) else "Final",
            }
        )

    if not candidates:
        return pd.DataFrame()

    verdict_df = pd.DataFrame(candidates)
    verdict_df["abs_score"] = verdict_df["score"].abs()
    verdict_df = verdict_df.sort_values(
        ["abs_score", "rr_ratio"], ascending=[False, False])
    return verdict_df.drop(columns=["abs_score"])


def apply_position_sizing(
    verdict_df: pd.DataFrame,
    account_capital: float,
    risk_per_trade_pct: float,
    max_trades: int = 3,
) -> pd.DataFrame:
    """Add quantity and PnL planning columns based on fixed-risk sizing."""
    if verdict_df.empty:
        return verdict_df

    risk_budget = account_capital * (risk_per_trade_pct / 100.0)
    out_df = verdict_df.head(max_trades).copy()

    quantities = []
    capital_used = []
    expected_profit = []
    expected_loss = []

    for row in out_df.itertuples(index=False):
        entry = float(row.entry)
        stop_loss = float(row.stop_loss)
        target = float(row.target)
        risk_per_share = abs(entry - stop_loss)

        if risk_per_share <= 0 or entry <= 0:
            qty = 0
        else:
            qty_by_risk = math.floor(risk_budget / risk_per_share)
            qty_by_capital = math.floor(account_capital / entry)
            qty = max(min(qty_by_risk, qty_by_capital), 0)

        trade_capital = qty * entry
        target_pnl = qty * abs(target - entry)
        stop_pnl = qty * abs(entry - stop_loss)

        quantities.append(int(qty))
        capital_used.append(round(trade_capital, 2))
        expected_profit.append(round(target_pnl, 2))
        expected_loss.append(round(stop_pnl, 2))

    out_df["qty"] = quantities
    out_df["capital_used"] = capital_used
    out_df["expected_profit_at_target"] = expected_profit
    out_df["expected_loss_at_stop"] = expected_loss
    return out_df


def log_verdict_run(
    verdict_df: pd.DataFrame,
    policy_name: str,
    account_capital: float,
    risk_per_trade_pct: float,
) -> Path:
    """Append current verdict recommendations to audit CSV."""
    if verdict_df.empty:
        return VERDICT_LOG_PATH

    VERDICT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not VERDICT_LOG_PATH.exists()

    rows = verdict_df.copy()
    rows.insert(0, "generated_at",
                datetime.now().isoformat(timespec="seconds"))
    rows.insert(1, "policy", policy_name)
    rows.insert(2, "account_capital", round(account_capital, 2))
    rows.insert(3, "risk_per_trade_pct", round(risk_per_trade_pct, 2))

    fieldnames = list(rows.columns)
    with VERDICT_LOG_PATH.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows.to_dict(orient="records"):
            writer.writerow(row)

    return VERDICT_LOG_PATH


def main() -> None:
    st.set_page_config(page_title="NSE EOD Analyzer",
                       layout="wide", initial_sidebar_state="expanded")
    inject_modern_theme()

    st.markdown(
        """
        <div class="hero-card">
          <p class="hero-title">NSE Signal Studio</p>
          <p class="hero-sub">Latest EOD refresh, live news sentiment, NIFTY 50 pulse scanner, and backtest analytics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Controls")
    selected_name = st.sidebar.selectbox(
        "Select NIFTY 50 stock", list(NIFTY_50_STOCKS.keys()))
    selected_ticker = NIFTY_50_STOCKS[selected_name]
    days = st.sidebar.slider(
        "Lookback days", min_value=45, max_value=252, value=90, step=5)
    news_items = st.sidebar.slider(
        "Headlines to score", min_value=5, max_value=12, value=8)
    use_live_news = st.sidebar.toggle("Use live RSS news", value=True)
    finalized_only = st.sidebar.toggle(
        "Use finalized EOD bars only",
        value=False,
        help="Exclude provisional quote-patched rows from signal and scanner calculations.",
    )

    if st.sidebar.button("Refresh market data now"):
        st.cache_data.clear()
        st.rerun()

    signal_tab, pulse_tab, verdict_tab, backtest_tab = st.tabs(
        ["Signal Dashboard", "NIFTY 50 Pulse", "Verdict", "Backtest"])

    with signal_tab:
        eod_df = get_eod_cached(selected_ticker, days=days)
        if eod_df.empty:
            st.error("No EOD data available for selected ticker.")
            return

        if finalized_only and "IS_PROVISIONAL" in eod_df.columns:
            finalized_df = eod_df[~eod_df["IS_PROVISIONAL"]].copy()
            if not finalized_df.empty:
                eod_df = finalized_df
            else:
                st.warning(
                    "No finalized rows are available yet for this symbol, so the latest available data is shown instead.")

        latest = eod_df.iloc[-1]
        close_value = _safe_float(latest.get("Close"), 0.0)
        prev_close = _safe_float(
            eod_df.iloc[-2].get("Close"), close_value) if len(eod_df) >= 2 else close_value
        day_change_pct = ((close_value - prev_close) /
                          prev_close) * 100 if prev_close > 0 else 0.0
        rsi_value = _safe_float(latest.get("RSI_14"), 50.0)
        macd_value = _safe_float(latest.get("MACD"), 0.0)
        macd_signal_value = _safe_float(latest.get("MACD_SIGNAL"), 0.0)
        sma_20_value = _safe_float(latest.get("SMA_20"), close_value)
        volatility_20 = float(eod_df["DAILY_RETURN_PCT"].tail(
            20).std()) if "DAILY_RETURN_PCT" in eod_df else 0.0
        latest_date = pd.to_datetime(eod_df.index[-1]).date()
        staleness_days = (datetime.utcnow().date() - latest_date).days
        latest_source = str(latest.get("DATA_SOURCE", "yfinance"))
        is_provisional = bool(latest.get("IS_PROVISIONAL", False))
        bar_status = "Provisional" if is_provisional else "Final"

        metric_cols = st.columns(6)
        metric_cols[0].metric("Close", f"{close_value:.2f}")
        metric_cols[1].metric("1D Move", f"{day_change_pct:.2f}%")
        metric_cols[2].metric("RSI (14)", f"{rsi_value:.2f}")
        metric_cols[3].metric(
            "MACD Spread", f"{(macd_value - macd_signal_value):.3f}")
        metric_cols[4].metric("Data Lag", f"{staleness_days} day(s)")
        metric_cols[5].metric("Bar Status", bar_status)

        st.caption(f"Data source: {latest_source}")

        if staleness_days > 1:
            st.warning(
                "Latest close appears stale. Use the refresh button or check if market was closed.")
        if is_provisional:
            st.warning(
                "Latest row is provisional (quote-patched) and may be revised when official EOD settles.")

        chart_df = eod_df[["Close", "SMA_20", "EMA_20", "SMA_50"]].copy()
        st.subheader(f"Price Structure: {selected_name} ({selected_ticker})")
        st.line_chart(chart_df, use_container_width=True)

        vcol1, vcol2 = st.columns([2, 1])
        with vcol1:
            st.subheader("Daily Volume")
            if "Volume" in eod_df.columns:
                st.bar_chart(eod_df["Volume"], use_container_width=True)
            else:
                st.info("Volume data not available.")
        with vcol2:
            st.subheader("Volatility Snapshot")
            st.metric("20D Std Dev (%)", f"{volatility_20:.2f}")
            st.metric(
                "20D SMA", f"{_safe_float(latest.get('SMA_20'), 0.0):.2f}")
            st.metric(
                "50D SMA", f"{_safe_float(latest.get('SMA_50'), 0.0):.2f}")

        support_df, resistance_df = _calculate_support_resistance_levels(
            eod_df,
            close_value,
            lookback_days=min(len(eod_df), 120),
            pivot_window=3,
            max_levels=3,
        )

        st.subheader("Support & Resistance Map")
        st.caption(
            "Derived from recent swing highs/lows, grouped into ATR-aware price bands, then ranked by touches and recency."
        )

        if support_df.empty and resistance_df.empty:
            st.info("Not enough repeated swing points yet to estimate stable support and resistance levels.")
        else:
            summary_cols = st.columns(4)
            top_support = support_df.iloc[0] if not support_df.empty else None
            top_resistance = resistance_df.iloc[0] if not resistance_df.empty else None

            summary_cols[0].metric(
                "Nearest Support",
                f"{float(top_support['Level']):.2f}" if top_support is not None else "N/A",
                f"{float(top_support['Distance to Close (%)']):.2f}% below" if top_support is not None else "",
            )
            summary_cols[1].metric(
                "Support Strength",
                f"{float(top_support['Strength']):.2f}" if top_support is not None else "N/A",
                f"{int(top_support['Touches'])} touch(es)" if top_support is not None else "",
            )
            summary_cols[2].metric(
                "Nearest Resistance",
                f"{float(top_resistance['Level']):.2f}" if top_resistance is not None else "N/A",
                f"{float(top_resistance['Distance to Close (%)']):.2f}% above" if top_resistance is not None else "",
            )
            summary_cols[3].metric(
                "Resistance Strength",
                f"{float(top_resistance['Strength']):.2f}" if top_resistance is not None else "N/A",
                f"{int(top_resistance['Touches'])} touch(es)" if top_resistance is not None else "",
            )

            level_frames = []
            if not support_df.empty:
                level_frames.append(support_df)
            if not resistance_df.empty:
                level_frames.append(resistance_df)

            if level_frames:
                levels_df = pd.concat(level_frames, ignore_index=True)
                levels_df = levels_df.sort_values(
                    ["Level Type", "Distance to Close (%)"],
                    ascending=[True, True],
                )
                st.dataframe(
                    levels_df.style.format(
                        {
                            "Level": "{:.2f}",
                            "Distance to Close (%)": "{:.2f}",
                            "Strength": "{:.2f}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            with st.expander("How the levels are derived"):
                st.markdown(
                    """
                    The dashboard uses a recent lookback window of daily OHLC bars, finds swing highs and lows with a centered pivot filter, clusters nearby prices using an ATR-aware tolerance band, and then ranks each cluster by the number of touches and how recently it was respected.
                    """
                )

        indicator_df = (
            latest[["Close", "RSI_14", "MACD", "MACD_SIGNAL",
                    "MACD_HIST", "SMA_20", "SMA_50", "EMA_20"]]
            .rename(
                {
                    "Close": "Close",
                    "RSI_14": "RSI (14)",
                    "MACD": "MACD",
                    "MACD_SIGNAL": "MACD Signal",
                    "MACD_HIST": "MACD Histogram",
                    "SMA_20": "SMA (20)",
                    "SMA_50": "SMA (50)",
                    "EMA_20": "EMA (20)",
                }
            )
            .to_frame("Value")
        )
        st.subheader("Latest Technical Snapshot")
        st.dataframe(indicator_df.style.format(
            {"Value": "{:.3f}"}), use_container_width=True)

        news_df = get_news_cached(
            selected_ticker, max_items=news_items, use_live_news=use_live_news)
        source = "RSS" if (not news_df.empty and str(
            news_df.iloc[0]["news_source"]) == "rss") else "Mock Fallback"
        st.subheader(f"News Sentiment ({source})")
        st.dataframe(news_df[["headline", "sentiment_score",
                     "sentiment_label"]], use_container_width=True)

        avg_sentiment = float(
            news_df["sentiment_score"].mean()) if not news_df.empty else 0.0
        recommendation, combined_score, technical_score, sentiment_score = generate_recommendation(
            rsi_value=rsi_value,
            macd_value=macd_value,
            macd_signal_value=macd_signal_value,
            close_value=close_value,
            sma_20_value=sma_20_value,
            avg_news_sentiment=avg_sentiment,
        )

        headline_count = int(len(news_df.index))
        confidence_band = derive_confidence_band(
            combined_score, headline_count)
        strength = min(max((abs(combined_score) / 1.0), 0.0), 1.0)

        st.subheader("Recommendation Box")
        st.info(
            (
                f"Recommendation: {recommendation}\n"
                f"Confidence: {confidence_band} | Signal strength: {strength * 100:.1f}%\n"
                f"Score blend: {combined_score:.3f} = 65% technical + 35% sentiment\n"
                f"Technical score: {technical_score:.3f} | Sentiment score: {sentiment_score:.3f}\n"
                f"Rule thresholds: BUY >= 0.25, SELL <= -0.25, else HOLD."
            )
        )
        st.progress(strength, text="Signal strength")

        if st.button("Log Recommendation Snapshot", use_container_width=False):
            log_path = log_recommendation_snapshot(
                ticker=selected_ticker,
                company_name=selected_name,
                recommendation=recommendation,
                confidence_band=confidence_band,
                combined_score=combined_score,
                technical_score=technical_score,
                sentiment_score=sentiment_score,
                rsi_value=rsi_value,
                macd_value=macd_value,
                macd_signal_value=macd_signal_value,
                headline_count=headline_count,
                news_source=source,
            )
            st.success(f"Saved snapshot to {log_path}")

    with pulse_tab:
        st.subheader("NIFTY 50 Pulse Scanner")
        st.caption(
            "Runs a cross-sectional technical scan on all configured NIFTY 50 stocks.")

        if st.button("Run NIFTY 50 scan", use_container_width=False):
            with st.spinner("Scanning NIFTY 50 components..."):
                pulse_df = build_nifty50_pulse(
                    days=days, finalized_only=finalized_only)

            if pulse_df.empty:
                st.warning("Pulse scanner could not fetch enough data.")
            else:
                top_buys = pulse_df[pulse_df["recommendation"] == "BUY"].head(
                    10)
                top_sells = pulse_df[pulse_df["recommendation"] == "SELL"].head(
                    10)

                c1, c2, c3 = st.columns(3)
                c1.metric("Stocks scanned", f"{len(pulse_df.index)}")
                c2.metric(
                    "BUY signals", f"{int((pulse_df['recommendation'] == 'BUY').sum())}")
                c3.metric(
                    "SELL signals", f"{int((pulse_df['recommendation'] == 'SELL').sum())}")

                st.markdown("### Top BUY Candidates")
                st.dataframe(top_buys, use_container_width=True)

                st.markdown("### Top SELL Pressure")
                st.dataframe(top_sells, use_container_width=True)

                st.markdown("### Full NIFTY 50 Pulse")
                st.dataframe(pulse_df, use_container_width=True)
        else:
            st.info("Click 'Run NIFTY 50 scan' to generate cross-stock signals.")

    with verdict_tab:
        st.subheader("Verdict: Next Session Top 3 Trades")
        st.caption(
            "Engine ranks NIFTY 50 stocks using RSI, MACD spread, trend regime, momentum, and ATR-based risk setup.")

        controls = st.columns(3)
        verdict_policy = controls[0].selectbox(
            "Policy", list(VERDICT_POLICIES.keys()), index=1)
        account_capital = controls[1].number_input(
            "Account Capital", min_value=50000.0, value=500000.0, step=10000.0
        )
        risk_per_trade_pct = controls[2].slider(
            "Risk per Trade (%)", min_value=0.25, max_value=3.00, value=1.00, step=0.25
        )

        if st.button("Generate Verdict", use_container_width=False):
            with st.spinner("Calculating ranked trade setups..."):
                verdict_df = build_verdict_candidates(
                    days=days,
                    finalized_only=finalized_only,
                    policy_name=verdict_policy,
                )

            verdict_df = apply_position_sizing(
                verdict_df=verdict_df,
                account_capital=float(account_capital),
                risk_per_trade_pct=float(risk_per_trade_pct),
                max_trades=3,
            )
            st.session_state["current_verdict_df"] = verdict_df

            if verdict_df.empty:
                st.warning(
                    "No high-conviction setups found for current settings.")
            else:
                total_capital_used = float(verdict_df["capital_used"].sum())
                total_expected_profit = float(
                    verdict_df["expected_profit_at_target"].sum())
                total_expected_loss = float(
                    verdict_df["expected_loss_at_stop"].sum())
                weighted_rr = (
                    total_expected_profit / total_expected_loss
                    if total_expected_loss > 0
                    else 0.0
                )

                summary_cols = st.columns(4)
                summary_cols[0].metric(
                    "Total Capital Used", f"{total_capital_used:.2f}")
                summary_cols[1].metric(
                    "Portfolio Loss @ Stops", f"{total_expected_loss:.2f}")
                summary_cols[2].metric(
                    "Portfolio Profit @ Targets", f"{total_expected_profit:.2f}")
                summary_cols[3].metric("Portfolio R:R", f"{weighted_rr:.2f}")

                for idx, row in verdict_df.iterrows():
                    st.markdown(
                        (
                            f"### {idx + 1}. {row['company']} ({row['ticker']}) - {row['side']}\n"
                            f"Entry: {row['entry']:.2f} | Target: {row['target']:.2f} | Stop Loss: {row['stop_loss']:.2f}\n"
                            f"Score: {row['score']:.3f} | R:R: {row['rr_ratio']:.2f} | "
                            f"RSI: {row['rsi_14']:.2f} | MACD Spread: {row['macd_spread']:.3f}\n"
                            f"Qty: {int(row['qty'])} | Capital Used: {row['capital_used']:.2f} | "
                            f"Profit@Target: {row['expected_profit_at_target']:.2f} | Loss@Stop: {row['expected_loss_at_stop']:.2f}\n"
                            f"Bar status: {row['bar_status']} | Source: {row['data_source']}"
                        )
                    )

                st.markdown("### Verdict Table")
                st.dataframe(verdict_df, use_container_width=True)

                csv_data = verdict_df.to_csv(index=False)
                st.download_button(
                    label="Download Verdict CSV",
                    data=csv_data,
                    file_name=f"verdict_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

                if st.button("Log Verdict Run", use_container_width=False):
                    log_path = log_verdict_run(
                        verdict_df=verdict_df,
                        policy_name=verdict_policy,
                        account_capital=float(account_capital),
                        risk_per_trade_pct=float(risk_per_trade_pct),
                    )
                    st.success(f"Saved verdict run to {log_path}")
        else:
            st.info(
                "Click 'Generate Verdict' to produce top 3 buy/sell setups for the next trading session.")

        cached_verdict = st.session_state.get("current_verdict_df")
        if isinstance(cached_verdict, pd.DataFrame) and not cached_verdict.empty:
            st.caption(
                "Last generated verdict is retained in session for quick export/logging.")

    with backtest_tab:
        st.subheader("Recommendation Backtest")
        log_df = load_recommendation_log()
        if log_df.empty:
            st.info(
                "No recommendation snapshots found yet. Log a few snapshots first.")
            return

        filter_cols = st.columns(2)
        ticker_filter = filter_cols[0].selectbox(
            "Filter ticker", ["All"] + sorted(log_df["ticker"].dropna().unique().tolist()))
        conf_filter = filter_cols[1].selectbox("Filter confidence", [
                                               "All"] + sorted(log_df["confidence"].dropna().unique().tolist()))

        if ticker_filter != "All":
            log_df = log_df[log_df["ticker"] == ticker_filter]
        if conf_filter != "All":
            log_df = log_df[log_df["confidence"] == conf_filter]

        backtest_df = build_backtest_results(log_df)
        if backtest_df.empty:
            st.warning(
                "No rows could be mapped to next-trading-day outcomes yet.")
            st.dataframe(log_df.tail(25), use_container_width=True)
            return

        total_signals = int(len(backtest_df.index))
        hit_rate = float(backtest_df["is_hit"].mean()
                         ) if total_signals else 0.0
        avg_strategy_return = float(
            backtest_df["strategy_return"].mean()) if total_signals else 0.0
        cumulative_return = float(
            (1 + backtest_df["strategy_return"]).prod() - 1) if total_signals else 0.0
        best_signal = backtest_df.loc[backtest_df["strategy_return"].idxmax()]

        metric_cols = st.columns(5)
        metric_cols[0].metric("Signals Backtested", f"{total_signals}")
        metric_cols[1].metric("Hit Rate", f"{hit_rate * 100:.1f}%")
        metric_cols[2].metric("Avg Strategy Return",
                              f"{avg_strategy_return * 100:.2f}%")
        metric_cols[3].metric("Cumulative Return",
                              f"{cumulative_return * 100:.2f}%")
        metric_cols[4].metric(
            "Best Signal", f"{best_signal['ticker']} ({best_signal['strategy_return'] * 100:.2f}%)")

        breakdown_df = (
            backtest_df.groupby("recommendation", as_index=False)
            .agg(
                signals=("recommendation", "count"),
                hit_rate=("is_hit", "mean"),
                avg_return=("strategy_return", "mean"),
            )
            .sort_values("recommendation")
        )
        breakdown_df["hit_rate"] = (breakdown_df["hit_rate"] * 100).round(2)
        breakdown_df["avg_return"] = (
            breakdown_df["avg_return"] * 100).round(2)

        st.markdown("### Performance by Recommendation")
        st.dataframe(breakdown_df, use_container_width=True)

        display_df = backtest_df.sort_values(
            "timestamp", ascending=False).copy()
        display_df["next_day_return"] = (
            display_df["next_day_return"] * 100).round(2)
        display_df["strategy_return"] = (
            display_df["strategy_return"] * 100).round(2)

        st.markdown("### Signal-Level Backtest Rows")
        st.dataframe(display_df, use_container_width=True)


if __name__ == "__main__":
    main()
