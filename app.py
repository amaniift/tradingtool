"""Streamlit dashboard for NSE EOD and news-sentiment analysis."""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.data.data_manager import (
    _YFINANCE_SESSION,
    fetch_eod_data,
    fetch_news_sentiment,
    fetch_nifty_option_chain,
    get_yfinance_session,
)


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
FEATURE_STORE_PATH = Path("data") / "feature_store.csv"
NIFTY_IV_HISTORY_PATH = Path("data") / "nifty_iv_history.csv"
HOLD_RETURN_BAND = 0.005

EVENT_DAY_RISK_TEMPLATES = {
    "Normal Session": {
        "max_risk_per_trade_pct": 1.0,
        "max_open_positions": 3,
        "stop_buffer_atr": 1.0,
        "target_buffer_atr": 1.8,
    },
    "Expiry Day": {
        "max_risk_per_trade_pct": 0.55,
        "max_open_positions": 2,
        "stop_buffer_atr": 0.8,
        "target_buffer_atr": 1.2,
    },
    "Macro Event Day": {
        "max_risk_per_trade_pct": 0.45,
        "max_open_positions": 1,
        "stop_buffer_atr": 0.7,
        "target_buffer_atr": 1.1,
    },
}

META_FEATURE_COLUMNS = [
    "rsi_14",
    "macd_spread",
    "sma20_gap_pct",
    "sma50_gap_pct",
    "ema20_gap_pct",
    "momentum_5d_pct",
    "volatility_20d_pct",
    "news_sentiment",
]

NIFTY_SECTOR_MAP = {
    "ADANIENT.NS": "Industrials",
    "ADANIPORTS.NS": "Industrials",
    "APOLLOHOSP.NS": "Healthcare",
    "ASIANPAINT.NS": "Consumer",
    "AXISBANK.NS": "Financials",
    "BAJAJ-AUTO.NS": "Auto",
    "BAJFINANCE.NS": "Financials",
    "BAJAJFINSV.NS": "Financials",
    "BEL.NS": "Industrials",
    "BHARTIARTL.NS": "Telecom",
    "CIPLA.NS": "Healthcare",
    "COALINDIA.NS": "Energy",
    "DRREDDY.NS": "Healthcare",
    "EICHERMOT.NS": "Auto",
    "ETERNAL.NS": "Consumer",
    "GRASIM.NS": "Materials",
    "HCLTECH.NS": "IT",
    "HDFCBANK.NS": "Financials",
    "HDFCLIFE.NS": "Financials",
    "HINDALCO.NS": "Materials",
    "HINDUNILVR.NS": "Consumer",
    "ICICIBANK.NS": "Financials",
    "INDIGO.NS": "Industrials",
    "INFY.NS": "IT",
    "ITC.NS": "Consumer",
    "JIOFIN.NS": "Financials",
    "JSWSTEEL.NS": "Materials",
    "KOTAKBANK.NS": "Financials",
    "LT.NS": "Industrials",
    "M&M.NS": "Auto",
    "MARUTI.NS": "Auto",
    "MAXHEALTH.NS": "Healthcare",
    "NESTLEIND.NS": "Consumer",
    "NTPC.NS": "Energy",
    "ONGC.NS": "Energy",
    "POWERGRID.NS": "Utilities",
    "RELIANCE.NS": "Energy",
    "SBILIFE.NS": "Financials",
    "SHRIRAMFIN.NS": "Financials",
    "SBIN.NS": "Financials",
    "SUNPHARMA.NS": "Healthcare",
    "TCS.NS": "IT",
    "TATACONSUM.NS": "Consumer",
    "TMPV.NS": "Auto",
    "TATASTEEL.NS": "Materials",
    "TECHM.NS": "IT",
    "TITAN.NS": "Consumer",
    "TRENT.NS": "Consumer",
    "ULTRACEMCO.NS": "Materials",
    "WIPRO.NS": "IT",
}

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
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

        :root {
            --bg: #06101f;
            --bg-2: #0b1630;
            --card: rgba(15, 23, 42, 0.78);
            --card-strong: rgba(17, 24, 39, 0.92);
            --line: rgba(148, 163, 184, 0.18);
            --text: #eff6ff;
            --muted: #9fb2cb;
            --soft: #6b7c93;
            --accent: #38bdf8;
            --accent-2: #34d399;
            --accent-3: #f59e0b;
            --danger: #f87171;
            --shadow: 0 22px 55px rgba(0, 0, 0, 0.35);
            --radius: 20px;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(56, 189, 248, 0.14), transparent 28%),
                radial-gradient(circle at top right, rgba(52, 211, 153, 0.10), transparent 24%),
                linear-gradient(180deg, var(--bg) 0%, #081224 38%, #060c18 100%);
            color: var(--text);
            font-family: 'Inter', sans-serif;
        }

        .block-container {
            max-width: 1440px;
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3, h4, h5 {
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: -0.03em;
            color: var(--text);
        }

        p, label, li, span, small, [data-testid="stMarkdownContainer"] {
            color: var(--muted);
        }

        .dashboard-shell {
            border: 1px solid var(--line);
            border-radius: 28px;
            background: linear-gradient(180deg, rgba(12, 18, 33, 0.84), rgba(8, 13, 24, 0.92));
            box-shadow: var(--shadow);
            padding: 1.35rem;
            margin-bottom: 1.2rem;
        }

        .hero-card {
            position: relative;
            overflow: hidden;
            background:
                linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(6, 10, 20, 0.96)),
                radial-gradient(circle at top right, rgba(56, 189, 248, 0.18), transparent 25%),
                radial-gradient(circle at bottom left, rgba(52, 211, 153, 0.12), transparent 20%);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 28px;
            padding: 1.6rem 1.75rem;
            box-shadow: var(--shadow);
            margin-bottom: 1.25rem;
        }

        .hero-card::after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.10), transparent 45%, rgba(52, 211, 153, 0.06));
            pointer-events: none;
        }

        .hero-title {
            font-size: 1.95rem;
            font-weight: 800;
            line-height: 1.05;
            margin: 0;
            color: var(--text);
        }

        .hero-accent {
            background: linear-gradient(135deg, #7dd3fc 0%, #34d399 55%, #fbbf24 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-sub {
            margin: 0.55rem 0 0;
            max-width: 68rem;
            color: var(--muted);
            font-size: 0.98rem;
            line-height: 1.55;
        }

        .hero-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1rem;
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.46rem 0.8rem;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(148, 163, 184, 0.18);
            color: #dbeafe;
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.01em;
        }

        .hero-badge strong {
            color: #ffffff;
            font-weight: 700;
        }

        .section-card {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: var(--radius);
            padding: 1rem 1.1rem 0.9rem;
            margin: 1rem 0;
            backdrop-filter: blur(14px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.18);
        }

        .section-card h3,
        .section-card h4 {
            margin-top: 0;
        }

        section[data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(7, 12, 24, 0.96), rgba(10, 18, 35, 0.96));
            border-right: 1px solid rgba(148, 163, 184, 0.16);
        }

        section[data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }

        section[data-testid="stSidebar"] [data-baseweb="select"] > div,
        section[data-testid="stSidebar"] [data-baseweb="base-input"] > div,
        section[data-testid="stSidebar"] .stButton > button,
        section[data-testid="stSidebar"] .stMultiSelect > div {
            background: rgba(15, 23, 42, 0.82);
            border: 1px solid rgba(148, 163, 184, 0.18);
            box-shadow: none;
            border-radius: 14px;
        }

        section[data-testid="stSidebar"] .stButton > button:hover {
            border-color: rgba(56, 189, 248, 0.55);
            background: rgba(8, 15, 28, 0.96);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.18);
            padding-bottom: 0.25rem;
        }

        .stTabs [data-baseweb="tab"] {
            color: var(--muted) !important;
            background: rgba(15, 23, 42, 0.55);
            border: 1px solid transparent;
            border-radius: 999px;
            padding: 0.72rem 1rem;
            font-weight: 700;
            font-family: 'Space Grotesk', sans-serif;
            transition: all 0.22s ease;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: #ffffff !important;
            border-color: rgba(56, 189, 248, 0.35);
            background: rgba(15, 23, 42, 0.9);
        }

        .stTabs [aria-selected="true"] {
            color: #ffffff !important;
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.22), rgba(52, 211, 153, 0.16));
            border: 1px solid rgba(56, 189, 248, 0.45) !important;
            box-shadow: 0 10px 20px rgba(8, 15, 28, 0.26);
        }

        [data-testid="stMetric"] {
            position: relative;
            background: linear-gradient(180deg, rgba(14, 22, 38, 0.86), rgba(8, 14, 26, 0.92));
            backdrop-filter: blur(14px);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.22);
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        }

        [data-testid="stMetric"]::before {
            content: "";
            position: absolute;
            inset: 0;
            border-radius: 18px;
            padding: 1px;
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.35), rgba(52, 211, 153, 0.12));
            -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
            -webkit-mask-composite: xor;
            mask-composite: exclude;
            pointer-events: none;
            opacity: 0.4;
        }

        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 18px 36px rgba(0, 0, 0, 0.30);
            border-color: rgba(56, 189, 248, 0.35);
        }

        [data-testid="stMetricLabel"] {
            color: var(--muted);
            font-weight: 600;
            font-size: 0.82rem;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }

        [data-testid="stMetricValue"] {
            color: var(--text);
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 1.42rem;
        }

        [data-testid="stMetricDelta"] {
            color: #cbd5e1;
        }

        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }

        ::-webkit-scrollbar-track {
            background: #081222;
        }

        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #334155, #1d4ed8);
            border-radius: 999px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, #475569, #2563eb);
        }

        .stButton > button {
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: linear-gradient(135deg, rgba(17, 24, 39, 0.92), rgba(15, 23, 42, 0.88));
            color: #f8fafc;
            font-weight: 700;
            padding: 0.62rem 1rem;
            transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        }

        .stButton > button:hover {
            border-color: rgba(56, 189, 248, 0.5);
            box-shadow: 0 12px 24px rgba(8, 15, 28, 0.35);
            transform: translateY(-1px);
        }

        [data-baseweb="base-input"] > div,
        [data-baseweb="select"] > div,
        .stNumberInput > div > div,
        .stTextInput > div > div,
        .stDateInput > div > div {
            background: rgba(11, 18, 32, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 14px;
            color: #f8fafc;
        }

        .stSlider [data-baseweb="slider"] div[role="slider"] {
            border-color: var(--accent);
            background: var(--accent);
        }

        .stSlider [data-baseweb="slider"] div[data-testid="stTickBar"] {
            background-color: rgba(148, 163, 184, 0.2);
        }

        div[data-testid="stExpander"] {
            background: rgba(15, 23, 42, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 18px;
        }

        div[data-testid="stExpander"] summary {
            color: #ffffff;
            font-weight: 700;
        }

        div[data-testid="stExpander"] summary:hover {
            color: #7dd3fc;
        }

        .stAlert {
            background: rgba(15, 23, 42, 0.82);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 16px;
            color: #f8fafc;
        }

        .stCaption {
            color: var(--soft) !important;
            font-size: 0.89rem;
        }

        .stDataFrame, div[data-testid="stTable"] {
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 18px;
            overflow: hidden;
            box-shadow: 0 14px 28px rgba(0, 0, 0, 0.18);
        }

        .stDataFrame * {
            color: #dbeafe !important;
        }

        div[data-testid="stDataFrame"] > div {
            background: rgba(8, 14, 26, 0.8);
        }

        .section-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(148, 163, 184, 0.3), transparent);
            margin: 1rem 0;
        }

        .subtle-kicker {
            color: #7dd3fc;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.16em;
            text-transform: uppercase;
        }

        .panel-title {
            color: #ffffff;
            font-size: 1.05rem;
            font-weight: 700;
            margin: 0 0 0.25rem;
        }

        .panel-subtitle {
            color: var(--muted);
            margin: 0;
            line-height: 1.5;
        }

        @media (max-width: 900px) {
            .block-container {
                padding-top: 0.85rem;
                padding-left: 0.8rem;
                padding-right: 0.8rem;
            }

            .hero-card {
                padding: 1.1rem 1rem;
                border-radius: 22px;
            }

            .hero-title {
                font-size: 1.42rem;
            }

            .stTabs [data-baseweb="tab"] {
                font-size: 0.86rem;
                padding: 0.56rem 0.8rem;
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


def _clamp(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    """Clamp a numeric value into a bounded range."""
    return max(lower, min(upper, value))


def _tanh_score(value: float, scale: float) -> float:
    """Map a raw value to a smooth -1..1 score."""
    if scale <= 0:
        return 0.0
    return math.tanh(value / scale)


def _relative_gap(current: float, reference: float) -> float:
    """Return percentage gap vs a reference price."""
    if current <= 0 or reference <= 0:
        return 0.0
    return ((current - reference) / reference) * 100.0


def _continuous_signal_components(
    rsi_value: float,
    macd_value: float,
    macd_signal_value: float,
    close_value: float,
    sma_20_value: float,
    sma_50_value: float | None = None,
) -> dict[str, float]:
    """Build smooth, magnitude-aware signal components for recommendation logic."""
    rsi_component = _tanh_score(50.0 - rsi_value, 12.0)
    macd_scale = max(abs(close_value) * 0.0025, 0.75)
    macd_component = _tanh_score(macd_value - macd_signal_value, macd_scale)

    trend20_gap = _relative_gap(close_value, sma_20_value)
    trend20_component = _tanh_score(trend20_gap, 1.6)

    if sma_50_value is None:
        trend50_component = trend20_component * 0.55
    else:
        trend50_gap = _relative_gap(close_value, sma_50_value)
        trend50_component = _tanh_score(trend50_gap, 2.4)

    return {
        "rsi_component": _clamp(rsi_component),
        "macd_component": _clamp(macd_component),
        "trend20_component": _clamp(trend20_component),
        "trend50_component": _clamp(trend50_component),
    }


def _render_price_structure_chart(
    eod_df: pd.DataFrame,
    selected_name: str,
    selected_ticker: str,
    selected_series: list[str],
) -> None:
    """Render a focused price chart with dynamic y-axis range."""
    if not selected_series:
        st.info("Select at least one series to render the chart.")
        return

    available_series = ["Close", "SMA_20", "EMA_20", "SMA_50"]
    requested_series = [series for series in selected_series if series in available_series]
    if not requested_series:
        st.info("Selected series are unavailable for this chart.")
        return

    chart_df = eod_df[requested_series].copy().dropna(how="all")
    if chart_df.empty:
        st.info("Price chart is unavailable for this symbol.")
        return

    chart_df = chart_df.reset_index().rename(columns={"index": "Date"})
    long_df = chart_df.melt(
        id_vars=["Date"],
        value_vars=requested_series,
        var_name="Series",
        value_name="Price",
    ).dropna(subset=["Price"])

    if long_df.empty:
        st.info("Price chart is unavailable for this symbol.")
        return

    y_min = float(long_df["Price"].min())
    y_max = float(long_df["Price"].max())
    value_range = max(y_max - y_min, 1.0)
    padding = max(value_range * 0.06, y_max * 0.003)
    y_domain = [max(0.0, y_min - padding), y_max + padding]

    st.subheader(f"Price Structure: {selected_name} ({selected_ticker})")
    st.caption(f"Focused y-axis range: {y_domain[0]:.2f} to {y_domain[1]:.2f}")

    series_colors = {
        "Close": "#74c0fc",
        "SMA_20": "#f97316",
        "EMA_20": "#38bdf8",
        "SMA_50": "#ef4444",
    }
    color_scale = alt.Scale(
        domain=requested_series,
        range=[series_colors[series] for series in requested_series],
    )

    chart = (
        alt.Chart(long_df)
        .mark_line(strokeWidth=2.2)
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y(
                "Price:Q",
                title="Price",
                scale=alt.Scale(domain=y_domain, zero=False),
            ),
            color=alt.Color("Series:N", scale=color_scale, legend=alt.Legend(orient="top")),
            tooltip=[
                alt.Tooltip("Date:T", title="Date"),
                alt.Tooltip("Series:N", title="Series"),
                alt.Tooltip("Price:Q", title="Price", format=",.2f"),
            ],
        )
        .properties(height=320)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


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
    components = _continuous_signal_components(
        rsi_value=rsi_value,
        macd_value=macd_value,
        macd_signal_value=macd_signal_value,
        close_value=close_value,
        sma_20_value=sma_20_value,
    )

    technical_score = (
        0.34 * components["rsi_component"]
        + 0.38 * components["macd_component"]
        + 0.28 * components["trend20_component"]
    )
    sentiment_score = _clamp(avg_news_sentiment)
    combined_score = (0.72 * technical_score) + (0.28 * sentiment_score)

    if combined_score >= 0.20:
        return "BUY", combined_score, technical_score, avg_news_sentiment
    if combined_score <= -0.20:
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


@st.cache_data(ttl=600, show_spinner=False)
def fetch_nifty50_index_data(days: int) -> pd.DataFrame:
    """Fetch NIFTY 50 index data with technical indicators."""
    import yfinance as yf

    index_ticker = "^NSEI"

    try:
        index_df = yf.download(
            index_ticker,
            start=(pd.Timestamp.today() - pd.Timedelta(days=days + 30)).strftime("%Y-%m-%d"),
            end=pd.Timestamp.today().strftime("%Y-%m-%d"),
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False,
            session=_YFINANCE_SESSION,
        )
    except Exception:
        return pd.DataFrame()

    if index_df.empty:
        return pd.DataFrame()

    if isinstance(index_df.columns, pd.MultiIndex):
        index_df.columns = index_df.columns.get_level_values(0)

    index_df = index_df.dropna(subset=["Close"])

    if "Volume" not in index_df.columns:
        index_df["Volume"] = 0

    close_series = index_df["Close"].astype(float)
    high_series = index_df["High"].astype(float) if "High" in index_df else close_series
    low_series = index_df["Low"].astype(float) if "Low" in index_df else close_series

    delta = close_series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    index_df["RSI_14"] = rsi

    ema12 = close_series.ewm(span=12, adjust=False).mean()
    ema26 = close_series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    index_df["MACD"] = macd
    index_df["MACD_SIGNAL"] = signal
    index_df["MACD_HIST"] = macd - signal

    sma20 = close_series.rolling(window=20).mean()
    sma50 = close_series.rolling(window=50).mean()
    ema20 = close_series.ewm(span=20, adjust=False).mean()
    index_df["SMA_20"] = sma20
    index_df["SMA_50"] = sma50
    index_df["EMA_20"] = ema20

    daily_return = close_series.pct_change() * 100
    index_df["DAILY_RETURN_PCT"] = daily_return

    return index_df


def calculate_market_breadth(df: pd.DataFrame) -> dict:
    """Calculate basic market breadth indicators from index data."""
    if df.empty or len(df) < 20:
        return {
            "advance_decline_ratio": "N/A",
            "new_highs_lows": "N/A",
            "percent_above_sma20": "N/A",
            "percent_above_sma50": "N/A",
            "trend_strength": "N/A",
        }

    close = df["Close"].astype(float)

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    above_sma20 = (close > sma20).sum()
    above_sma50 = (close > sma50).sum()
    total = len(close.dropna())

    pct_above_20 = (above_sma20 / total * 100) if total > 0 else 0
    pct_above_50 = (above_sma50 / total * 100) if total > 0 else 0

    recent_returns = df["DAILY_RETURN_PCT"].tail(5) if "DAILY_RETURN_PCT" in df else pd.Series()
    trend_strength = "Strong" if recent_returns.mean() > 0.5 else "Weak" if recent_returns.mean() < -0.5 else "Neutral"

    return {
        "percent_above_sma20": f"{pct_above_20:.1f}%",
        "percent_above_sma50": f"{pct_above_50:.1f}%",
        "trend_strength": trend_strength,
        "5day_momentum": f"{recent_returns.sum():.2f}%" if not recent_returns.empty else "N/A",
    }


def _calculate_next_session_pivots(index_df: pd.DataFrame) -> pd.DataFrame:
    """Compute classical pivot levels for the next trading session from latest OHLC."""
    required_cols = {"High", "Low", "Close"}
    if index_df.empty or not required_cols.issubset(index_df.columns):
        return pd.DataFrame()

    latest_bar = index_df.iloc[-1]
    high_value = _safe_float(latest_bar.get("High"), 0.0)
    low_value = _safe_float(latest_bar.get("Low"), 0.0)
    close_value = _safe_float(latest_bar.get("Close"), 0.0)

    if high_value <= 0 or low_value <= 0 or close_value <= 0:
        return pd.DataFrame()

    pivot = (high_value + low_value + close_value) / 3.0
    spread = high_value - low_value

    levels = [
        {"Level": "R3", "Price": high_value + 2.0 * (pivot - low_value)},
        {"Level": "R2", "Price": pivot + spread},
        {"Level": "R1", "Price": (2.0 * pivot) - low_value},
        {"Level": "Pivot", "Price": pivot},
        {"Level": "S1", "Price": (2.0 * pivot) - high_value},
        {"Level": "S2", "Price": pivot - spread},
        {"Level": "S3", "Price": low_value - 2.0 * (high_value - pivot)},
    ]

    pivot_df = pd.DataFrame(levels)
    pivot_df["Distance to Close (%)"] = ((pivot_df["Price"] - close_value) / close_value) * 100.0
    return pivot_df


def _softmax(values: list[float]) -> list[float]:
    """Compute softmax probabilities for small vectors."""
    if not values:
        return []
    max_v = max(values)
    exp_vals = [math.exp(v - max_v) for v in values]
    total = sum(exp_vals)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [val / total for val in exp_vals]


def _compute_derivatives_insights(snapshot: dict) -> tuple[dict[str, float | str], pd.DataFrame]:
    """Derive OI/PCR/max-pain/gamma and key strikes from option-chain snapshot."""
    chain_df = snapshot.get("chain_df")
    if not isinstance(chain_df, pd.DataFrame) or chain_df.empty:
        return {}, pd.DataFrame()

    df = chain_df.copy()
    for col in ["strike", "ce_oi", "pe_oi", "ce_iv", "pe_iv", "ce_gamma", "pe_gamma"]:
        if col not in df.columns:
            df[col] = 0.0
    df[["strike", "ce_oi", "pe_oi", "ce_iv", "pe_iv", "ce_gamma", "pe_gamma"]] = (
        df[["strike", "ce_oi", "pe_oi", "ce_iv", "pe_iv", "ce_gamma", "pe_gamma"]]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )

    underlying = _safe_float(snapshot.get("underlying_value"), 0.0)
    total_call_oi = float(df["ce_oi"].sum())
    total_put_oi = float(df["pe_oi"].sum())
    pcr = (total_put_oi / total_call_oi) if total_call_oi > 0 else 0.0

    if underlying > 0:
        atm_idx = (df["strike"] - underlying).abs().idxmin()
        atm_row = df.loc[atm_idx]
        iv_values = [v for v in [atm_row.get("ce_iv", 0.0), atm_row.get("pe_iv", 0.0)] if v and v > 0]
        atm_iv = float(sum(iv_values) / len(iv_values)) if iv_values else 0.0
    else:
        atm_iv = 0.0

    strikes = df["strike"].tolist()
    call_oi = df["ce_oi"].tolist()
    put_oi = df["pe_oi"].tolist()
    max_pain_strike = 0.0
    min_pain = None
    for settlement in strikes:
        pain = 0.0
        for strike, coi, poi in zip(strikes, call_oi, put_oi):
            pain += max(0.0, settlement - strike) * coi
            pain += max(0.0, strike - settlement) * poi
        if min_pain is None or pain < min_pain:
            min_pain = pain
            max_pain_strike = settlement

    df["gamma_exposure"] = (df["ce_gamma"] * df["ce_oi"]) + (df["pe_gamma"] * df["pe_oi"])
    gamma_wall = float(df.loc[df["gamma_exposure"].idxmax(), "strike"]) if not df.empty else 0.0
    df["total_oi"] = df["ce_oi"] + df["pe_oi"]
    magnets = df.sort_values(["total_oi", "gamma_exposure"], ascending=[False, False]).head(6).copy()
    if underlying > 0:
        magnets["distance_to_spot_pct"] = ((magnets["strike"] - underlying) / underlying) * 100.0
    else:
        magnets["distance_to_spot_pct"] = 0.0

    metrics = {
        "underlying": underlying,
        "pcr": pcr,
        "total_call_oi": total_call_oi,
        "total_put_oi": total_put_oi,
        "max_pain": float(max_pain_strike),
        "atm_iv": float(atm_iv),
        "gamma_wall": float(gamma_wall),
        "expiry": str(snapshot.get("nearest_expiry") or "N/A"),
    }
    return metrics, magnets[["strike", "total_oi", "gamma_exposure", "distance_to_spot_pct"]]


def _update_iv_history_and_percentile(expiry: str, atm_iv: float) -> float | None:
    """Persist ATM IV observations and return trailing IV percentile."""
    if atm_iv <= 0:
        return None

    NIFTY_IV_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.now().date().isoformat()
    new_row = pd.DataFrame(
        [
            {
                "as_of_date": today,
                "expiry": expiry,
                "atm_iv": float(atm_iv),
            }
        ]
    )

    if NIFTY_IV_HISTORY_PATH.exists():
        hist_df = pd.read_csv(NIFTY_IV_HISTORY_PATH)
    else:
        hist_df = pd.DataFrame(columns=["as_of_date", "expiry", "atm_iv"])

    hist_df = hist_df[~((hist_df["as_of_date"].astype(str) == today) & (hist_df["expiry"].astype(str) == expiry))]
    hist_df = pd.concat([hist_df, new_row], ignore_index=True)
    hist_df["atm_iv"] = pd.to_numeric(hist_df["atm_iv"], errors="coerce")
    hist_df = hist_df.dropna(subset=["atm_iv"])
    hist_df.to_csv(NIFTY_IV_HISTORY_PATH, index=False)

    if len(hist_df) < 5:
        return None
    percentile = float((hist_df["atm_iv"] <= atm_iv).mean() * 100.0)
    return percentile


def _days_to_expiry(expiry_text: str) -> int | None:
    """Parse NSE expiry date string and return days to expiry."""
    if not expiry_text or expiry_text == "N/A":
        return None
    parsed = pd.to_datetime(expiry_text, dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return int((parsed.date() - datetime.now().date()).days)


@st.cache_data(ttl=1800, show_spinner=False)
def build_feature_store_snapshot(days: int, finalized_only: bool, use_live_news: bool, max_news_items: int = 6) -> pd.DataFrame:
    """Build cross-sectional feature store for all NIFTY 50 names."""
    import yfinance as yf

    rows = []
    for company, ticker in NIFTY_50_STOCKS.items():
        eod_df = fetch_eod_data(ticker, days=max(days, 130))
        if eod_df.empty:
            continue

        if finalized_only and "IS_PROVISIONAL" in eod_df.columns:
            finalized_df = eod_df[~eod_df["IS_PROVISIONAL"]].copy()
            if not finalized_df.empty:
                eod_df = finalized_df

        if len(eod_df) < 60:
            continue

        latest = eod_df.iloc[-1]
        close_value = _safe_float(latest.get("Close"), 0.0)
        if close_value <= 0:
            continue

        sma20 = _safe_float(latest.get("SMA_20"), close_value)
        sma50 = _safe_float(latest.get("SMA_50"), close_value)
        ema20 = _safe_float(latest.get("EMA_20"), close_value)
        macd = _safe_float(latest.get("MACD"), 0.0)
        macd_signal = _safe_float(latest.get("MACD_SIGNAL"), 0.0)
        rsi = _safe_float(latest.get("RSI_14"), 50.0)
        daily_return_std = float(eod_df["DAILY_RETURN_PCT"].tail(20).std()) if "DAILY_RETURN_PCT" in eod_df else 0.0
        momentum_5d = float(eod_df["Close"].pct_change(5).iloc[-1] * 100.0)
        next_day_target = float(eod_df["Close"].pct_change().shift(-1).iloc[-2] * 100.0) if len(eod_df) > 6 else 0.0

        vol_mean = float(eod_df["Volume"].tail(20).mean()) if "Volume" in eod_df else 0.0
        vol_std = float(eod_df["Volume"].tail(20).std()) if "Volume" in eod_df else 0.0
        latest_vol = _safe_float(latest.get("Volume"), 0.0)
        volume_z = ((latest_vol - vol_mean) / vol_std) if vol_std > 0 else 0.0

        news_df = fetch_news_sentiment(ticker, max_items=max_news_items, use_live_news=use_live_news)
        news_sentiment = float(news_df["sentiment_score"].mean()) if not news_df.empty else 0.0

        recommendation, combined_score, technical_score, _ = generate_recommendation(
            rsi_value=rsi,
            macd_value=macd,
            macd_signal_value=macd_signal,
            close_value=close_value,
            sma_20_value=sma20,
            avg_news_sentiment=news_sentiment,
        )

        trailing_pe = 0.0
        market_cap_cr = 0.0
        beta = 0.0
        try:
            info = yf.Ticker(ticker, session=get_yfinance_session()).fast_info
            if hasattr(info, "get"):
                trailing_pe = _safe_float(info.get("trailingPE"), 0.0)
                market_cap = _safe_float(info.get("marketCap"), 0.0)
                beta = _safe_float(info.get("beta"), 0.0)
                market_cap_cr = (market_cap / 1e7) if market_cap > 0 else 0.0
        except Exception:
            pass

        rows.append(
            {
                "as_of": datetime.now().isoformat(timespec="seconds"),
                "company": company,
                "ticker": ticker,
                "close": round(close_value, 2),
                "rsi_14": round(rsi, 3),
                "macd_spread": round(macd - macd_signal, 4),
                "sma20_gap_pct": round(((close_value - sma20) / sma20) * 100.0 if sma20 > 0 else 0.0, 4),
                "sma50_gap_pct": round(((close_value - sma50) / sma50) * 100.0 if sma50 > 0 else 0.0, 4),
                "ema20_gap_pct": round(((close_value - ema20) / ema20) * 100.0 if ema20 > 0 else 0.0, 4),
                "momentum_5d_pct": round(momentum_5d, 4),
                "volatility_20d_pct": round(daily_return_std, 4),
                "volume_zscore": round(volume_z, 4),
                "news_sentiment": round(news_sentiment, 4),
                "rule_recommendation": recommendation,
                "rule_score": round(combined_score, 4),
                "technical_score": round(technical_score, 4),
                "trailing_pe": round(trailing_pe, 4),
                "market_cap_cr": round(market_cap_cr, 2),
                "beta": round(beta, 4),
                "next_day_return_pct_target": round(next_day_target, 4),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _persist_feature_store_snapshot(feature_df: pd.DataFrame) -> Path:
    """Append latest feature snapshot to persistent CSV store."""
    FEATURE_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not FEATURE_STORE_PATH.exists()
    mode = "a"
    feature_df.to_csv(FEATURE_STORE_PATH, mode=mode, header=write_header, index=False)
    return FEATURE_STORE_PATH


def _fit_meta_linear_model(feature_df: pd.DataFrame) -> tuple[np.ndarray, pd.Series, pd.Series] | None:
    """Fit a lightweight linear model over engineered features to estimate return."""
    required_cols = META_FEATURE_COLUMNS + ["next_day_return_pct_target"]
    if feature_df.empty or not set(required_cols).issubset(feature_df.columns):
        return None

    train_df = feature_df[required_cols].copy().dropna()
    if len(train_df) < 25:
        return None

    x = train_df[META_FEATURE_COLUMNS].astype(float)
    y = train_df["next_day_return_pct_target"].astype(float)

    mu = x.mean()
    sigma = x.std().replace(0.0, 1.0)
    x_scaled = (x - mu) / sigma
    x_mat = np.column_stack([np.ones(len(x_scaled.index)), x_scaled.values])

    ridge_lambda = 0.8
    ridge_eye = np.eye(x_mat.shape[1])
    ridge_eye[0, 0] = 0.0
    beta = np.linalg.pinv(x_mat.T @ x_mat + ridge_lambda * ridge_eye) @ x_mat.T @ y.values
    return beta, mu, sigma


def _build_probabilistic_outlook(rule_score: float, ml_return_pct: float, volatility_pct: float) -> dict[str, float]:
    """Convert blended signals into a simple return-probability distribution."""
    ml_score = max(min(ml_return_pct / 1.5, 2.5), -2.5)
    vol_penalty = min(max(volatility_pct / 2.5, 0.0), 1.2)
    up_logit = (1.1 * rule_score) + (0.75 * ml_score) - (0.25 * vol_penalty)
    down_logit = (-1.1 * rule_score) + (-0.75 * ml_score) - (0.25 * vol_penalty)
    flat_logit = 0.35 + (0.45 * vol_penalty) - (0.25 * abs(rule_score))

    p_up, p_down, p_flat = _softmax([up_logit, down_logit, flat_logit])
    expected_return_pct = (p_up - p_down) * max(0.35, volatility_pct * 0.9)
    return {
        "prob_up": p_up,
        "prob_down": p_down,
        "prob_flat": p_flat,
        "expected_return_pct": expected_return_pct,
    }


def run_meta_model_blend(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Blend rule-engine output with an ML ranking score and probabilities."""
    if feature_df.empty:
        return pd.DataFrame()

    out = feature_df.copy()
    model = _fit_meta_linear_model(out)
    if model is None:
        out["ml_expected_return_pct"] = 0.0
    else:
        beta, mu, sigma = model
        x_live = out[META_FEATURE_COLUMNS].astype(float)
        x_live_scaled = (x_live - mu) / sigma
        x_live_mat = np.column_stack([np.ones(len(x_live_scaled.index)), x_live_scaled.values])
        out["ml_expected_return_pct"] = x_live_mat @ beta

    out["ml_score"] = out["ml_expected_return_pct"].apply(lambda v: max(min(v / 1.6, 1.0), -1.0))
    out["meta_score"] = (0.58 * out["rule_score"]) + (0.42 * out["ml_score"])

    prob_rows = []
    for row in out.itertuples(index=False):
        probs = _build_probabilistic_outlook(
            rule_score=float(getattr(row, "rule_score", 0.0)),
            ml_return_pct=float(getattr(row, "ml_expected_return_pct", 0.0)),
            volatility_pct=abs(float(getattr(row, "volatility_20d_pct", 0.0))),
        )
        prob_rows.append(probs)

    prob_df = pd.DataFrame(prob_rows)
    out = pd.concat([out.reset_index(drop=True), prob_df], axis=1)

    out["meta_recommendation"] = out["meta_score"].apply(
        lambda s: "BUY" if s >= 0.20 else "SELL" if s <= -0.20 else "HOLD"
    )
    out = out.sort_values(["meta_score", "ml_expected_return_pct"], ascending=[False, False]).reset_index(drop=True)
    return out


@st.cache_data(ttl=300, show_spinner=False)
def fetch_intraday_data(ticker: str, interval: str = "5m", period_days: int = 5) -> pd.DataFrame:
    """Fetch intraday OHLCV bars for NSE symbols using yfinance."""
    import yfinance as yf

    safe_days = min(max(int(period_days), 1), 30)
    period = f"{safe_days}d"

    try:
        intraday_df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False,
            session=get_yfinance_session(),
        )
    except Exception:
        return pd.DataFrame()

    if intraday_df.empty:
        return pd.DataFrame()

    if isinstance(intraday_df.columns, pd.MultiIndex):
        intraday_df.columns = intraday_df.columns.get_level_values(0)

    out = intraday_df.copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out


def _compute_opening_range_stats(intraday_df: pd.DataFrame, opening_range_mins: int) -> dict[str, float | str]:
    """Compute opening range, breakout direction, and failure flags for latest session."""
    if intraday_df.empty or "Open" not in intraday_df.columns:
        return {}

    latest_date = intraday_df.index[-1].date()
    session_df = intraday_df[intraday_df.index.date == latest_date].copy()
    if session_df.empty:
        return {}

    if len(session_df.index) >= 2:
        interval_minutes = int(max((session_df.index[1] - session_df.index[0]).total_seconds() // 60, 1))
    else:
        interval_minutes = 5

    bars_for_or = max(int(opening_range_mins // interval_minutes), 1)
    opening_slice = session_df.head(bars_for_or)
    if opening_slice.empty:
        return {}

    or_high = float(opening_slice["High"].max()) if "High" in opening_slice else 0.0
    or_low = float(opening_slice["Low"].min()) if "Low" in opening_slice else 0.0
    current_close = float(session_df["Close"].iloc[-1]) if "Close" in session_df else 0.0

    broke_above = bool((session_df.get("High", pd.Series(dtype=float)) > or_high).any())
    broke_below = bool((session_df.get("Low", pd.Series(dtype=float)) < or_low).any())
    failed_above = broke_above and (current_close < or_high)
    failed_below = broke_below and (current_close > or_low)

    if current_close > or_high:
        bias = "Bullish Breakout"
    elif current_close < or_low:
        bias = "Bearish Breakdown"
    else:
        bias = "Inside Range"

    return {
        "session_date": latest_date.isoformat(),
        "opening_high": or_high,
        "opening_low": or_low,
        "current_close": current_close,
        "opening_range_pct": ((or_high - or_low) / current_close) * 100.0 if current_close > 0 else 0.0,
        "breakout_bias": bias,
        "breakout_failed_up": int(failed_above),
        "breakout_failed_down": int(failed_below),
        "bars_in_session": int(len(session_df.index)),
    }


def _compute_intraday_pivot_reactions(intraday_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Measure intraday pivot touches and breakout-failure counts for latest session."""
    if intraday_df.empty or not {"High", "Low", "Close"}.issubset(intraday_df.columns):
        return pd.DataFrame(), {}

    daily_df = (
        intraday_df[["Open", "High", "Low", "Close"]]
        .resample("1D")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
        .dropna(subset=["Close"])
    )
    if len(daily_df.index) < 2:
        return pd.DataFrame(), {}

    prev_day = daily_df.iloc[-2]
    prev_ohlc = pd.DataFrame([prev_day])[ ["High", "Low", "Close"] ]
    pivot_df = _calculate_next_session_pivots(prev_ohlc)
    if pivot_df.empty:
        return pd.DataFrame(), {}

    latest_date = intraday_df.index[-1].date()
    session_df = intraday_df[intraday_df.index.date == latest_date].copy()
    if session_df.empty:
        return pivot_df, {}

    spot = float(session_df["Close"].iloc[-1])
    tolerance = max(spot * 0.0008, 0.5)
    touch_rows = []
    for row in pivot_df.itertuples(index=False):
        level_name = str(row.Level)
        level_value = float(row.Price)
        touches = int(((session_df["Low"] <= level_value + tolerance) & (session_df["High"] >= level_value - tolerance)).sum())
        touch_rows.append({"Level": level_name, "Price": level_value, "Touches": touches})

    touch_df = pd.DataFrame(touch_rows)
    level_map = {str(row.Level): float(row.Price) for row in pivot_df.itertuples(index=False)}
    r1 = level_map.get("R1")
    s1 = level_map.get("S1")
    failed_r1_breaks = int(((session_df["High"] > r1) & (session_df["Close"] < r1)).sum()) if r1 is not None else 0
    failed_s1_breaks = int(((session_df["Low"] < s1) & (session_df["Close"] > s1)).sum()) if s1 is not None else 0

    stats = {
        "failed_r1_breaks": float(failed_r1_breaks),
        "failed_s1_breaks": float(failed_s1_breaks),
        "pivot_touch_total": float(touch_df["Touches"].sum()) if not touch_df.empty else 0.0,
    }
    return touch_df, stats


def _build_preopen_gap_playbook(ticker: str, intraday_df: pd.DataFrame) -> dict[str, float | str]:
    """Create a pre-open gap and opening-auction style playbook for latest session."""
    if intraday_df.empty:
        return {}

    latest_date = intraday_df.index[-1].date()
    session_df = intraday_df[intraday_df.index.date == latest_date].copy()
    if session_df.empty:
        return {}

    daily_ref = fetch_eod_data(ticker, days=3)
    if daily_ref.empty:
        return {}
    prev_close = _safe_float(daily_ref.iloc[-1].get("Close"), 0.0)
    first_open = _safe_float(session_df.iloc[0].get("Open"), 0.0)
    if prev_close <= 0 or first_open <= 0:
        return {}

    gap_pct = ((first_open - prev_close) / prev_close) * 100.0
    opening_slice = session_df.head(min(3, len(session_df.index)))
    auction_return = ((_safe_float(opening_slice.iloc[-1].get("Close"), first_open) - first_open) / first_open) * 100.0
    first_volume = float(opening_slice.get("Volume", pd.Series(dtype=float)).sum()) if "Volume" in opening_slice else 0.0
    median_volume = float(intraday_df.get("Volume", pd.Series(dtype=float)).median()) if "Volume" in intraday_df else 0.0
    volume_ratio = (first_volume / median_volume) if median_volume > 0 else 1.0

    if gap_pct >= 0.7 and auction_return > 0:
        setup = "Gap-up continuation candidate"
    elif gap_pct >= 0.7 and auction_return < 0:
        setup = "Gap-up fade candidate"
    elif gap_pct <= -0.7 and auction_return < 0:
        setup = "Gap-down continuation candidate"
    elif gap_pct <= -0.7 and auction_return > 0:
        setup = "Gap-down mean-reversion candidate"
    else:
        setup = "Neutral open; wait for opening range break"

    auction_signal = "Strong" if abs(auction_return) >= 0.25 and volume_ratio >= 3.0 else "Moderate" if abs(auction_return) >= 0.12 else "Weak"
    return {
        "prev_close": prev_close,
        "session_open": first_open,
        "gap_pct": gap_pct,
        "auction_return_pct": auction_return,
        "auction_volume_ratio": volume_ratio,
        "playbook": setup,
        "auction_signal": auction_signal,
    }


@st.cache_data(ttl=1200, show_spinner=False)
def _build_portfolio_correlations(tickers: tuple[str, ...], lookback_days: int, finalized_only: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build return matrix and correlation matrix for selected tickers."""
    returns_map: dict[str, pd.Series] = {}
    for ticker in tickers:
        eod_df = fetch_eod_data(ticker, days=max(lookback_days, 90))
        if eod_df.empty or "Close" not in eod_df.columns:
            continue
        if finalized_only and "IS_PROVISIONAL" in eod_df.columns:
            final_df = eod_df[~eod_df["IS_PROVISIONAL"]].copy()
            if not final_df.empty:
                eod_df = final_df
        series = eod_df["Close"].astype(float).pct_change().dropna()
        if not series.empty:
            returns_map[ticker] = series.tail(lookback_days)

    if not returns_map:
        return pd.DataFrame(), pd.DataFrame()

    returns_df = pd.DataFrame(returns_map).dropna(how="all")
    if returns_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    corr_df = returns_df.corr().fillna(0.0)
    return returns_df, corr_df


def build_portfolio_from_verdict(
    verdict_df: pd.DataFrame,
    account_capital: float,
    max_positions: int,
    max_sector_exposure_pct: float,
    lookback_days: int,
    drawdown_guardrail_pct: float,
    finalized_only: bool,
) -> dict[str, object]:
    """Construct a multi-position portfolio with correlation and sector constraints."""
    if verdict_df.empty:
        return {}

    base_df = verdict_df.copy().head(max_positions).reset_index(drop=True)
    if base_df.empty:
        return {}

    base_df["sector"] = base_df["ticker"].map(NIFTY_SECTOR_MAP).fillna("Other")
    returns_df, corr_df = _build_portfolio_correlations(tuple(base_df["ticker"].tolist()), lookback_days, finalized_only)

    raw_scores: list[float] = []
    avg_corr_list: list[float] = []
    for row in base_df.itertuples(index=False):
        ticker = str(row.ticker)
        if not corr_df.empty and ticker in corr_df.index:
            peers = corr_df.loc[ticker].drop(labels=[ticker], errors="ignore")
            avg_corr = float(peers.mean()) if not peers.empty else 0.0
        else:
            avg_corr = 0.0
        avg_corr_list.append(avg_corr)

        quality = abs(float(getattr(row, "score", 0.0))) * max(float(getattr(row, "rr_ratio", 1.0)), 1.0)
        diversifier = 1.0 / (1.0 + max(avg_corr, 0.0))
        raw_scores.append(max(quality * diversifier, 0.01))

    raw_total = sum(raw_scores)
    prelim_weights = [score / raw_total for score in raw_scores] if raw_total > 0 else [1.0 / len(raw_scores)] * len(raw_scores)

    sector_cap = max_sector_exposure_pct / 100.0
    sector_used: dict[str, float] = {}
    weights = [0.0] * len(prelim_weights)
    remaining_total = 1.0

    ordering = sorted(range(len(prelim_weights)), key=lambda idx: prelim_weights[idx], reverse=True)
    for idx in ordering:
        sector = str(base_df.at[idx, "sector"])
        used = sector_used.get(sector, 0.0)
        headroom = max(sector_cap - used, 0.0)
        alloc = min(prelim_weights[idx], headroom, remaining_total)
        weights[idx] = max(alloc, 0.0)
        sector_used[sector] = used + weights[idx]
        remaining_total -= weights[idx]

    if remaining_total > 1e-6:
        eligible = [idx for idx in ordering if weights[idx] < prelim_weights[idx]]
        for idx in eligible:
            sector = str(base_df.at[idx, "sector"])
            headroom = max(sector_cap - sector_used.get(sector, 0.0), 0.0)
            extra = min(headroom, remaining_total)
            weights[idx] += extra
            sector_used[sector] = sector_used.get(sector, 0.0) + extra
            remaining_total -= extra
            if remaining_total <= 1e-6:
                break

    weight_sum = sum(weights)
    if weight_sum > 0:
        weights = [w / weight_sum for w in weights]

    base_df["avg_corr"] = [round(v, 3) for v in avg_corr_list]
    base_df["alloc_weight"] = weights
    base_df["capital_alloc"] = [round(account_capital * w, 2) for w in weights]

    qty_alloc = []
    capital_used = []
    expected_profit = []
    expected_loss = []
    for row in base_df.itertuples(index=False):
        entry = float(row.entry)
        stop_loss = float(row.stop_loss)
        target = float(row.target)
        alloc_cap = float(row.capital_alloc)
        if entry <= 0:
            qty = 0
        else:
            qty = max(math.floor(alloc_cap / entry), 0)
        qty_alloc.append(int(qty))
        capital_used.append(round(qty * entry, 2))
        expected_profit.append(round(qty * abs(target - entry), 2))
        expected_loss.append(round(qty * abs(entry - stop_loss), 2))

    base_df["qty_alloc"] = qty_alloc
    base_df["capital_used"] = capital_used
    base_df["expected_profit"] = expected_profit
    base_df["expected_loss"] = expected_loss

    if not returns_df.empty:
        signed_returns = returns_df.copy()
        for row in base_df.itertuples(index=False):
            ticker = str(row.ticker)
            if ticker not in signed_returns.columns:
                continue
            if str(row.side).upper() == "SELL":
                signed_returns[ticker] = -signed_returns[ticker]

        weighted = pd.Series(0.0, index=signed_returns.index)
        for row in base_df.itertuples(index=False):
            ticker = str(row.ticker)
            if ticker in signed_returns.columns:
                weighted = weighted + (float(row.alloc_weight) * signed_returns[ticker].fillna(0.0))
        weighted = weighted.dropna()
    else:
        weighted = pd.Series(dtype=float)

    if weighted.empty:
        var_95 = 0.0
        cvar_95 = 0.0
        max_drawdown_pct = 0.0
    else:
        q05 = float(np.quantile(weighted.values, 0.05))
        tail = weighted[weighted <= q05]
        var_95 = abs(q05) * 100.0
        cvar_95 = abs(float(tail.mean())) * 100.0 if not tail.empty else var_95
        equity = (1.0 + weighted).cumprod()
        drawdown = (equity / equity.cummax()) - 1.0
        max_drawdown_pct = abs(float(drawdown.min())) * 100.0 if not drawdown.empty else 0.0

    guardrail_breached = max_drawdown_pct > drawdown_guardrail_pct
    de_risk_factor = min(drawdown_guardrail_pct / max_drawdown_pct, 1.0) if max_drawdown_pct > 0 else 1.0

    sector_df = (
        base_df.groupby("sector", as_index=False)["alloc_weight"]
        .sum()
        .rename(columns={"alloc_weight": "weight"})
        .sort_values("weight", ascending=False)
    )
    sector_df["weight_pct"] = sector_df["weight"] * 100.0

    summary = {
        "positions": int(len(base_df.index)),
        "capital_used": float(base_df["capital_used"].sum()),
        "expected_profit": float(base_df["expected_profit"].sum()),
        "expected_loss": float(base_df["expected_loss"].sum()),
        "portfolio_rr": (float(base_df["expected_profit"].sum()) / float(base_df["expected_loss"].sum())) if float(base_df["expected_loss"].sum()) > 0 else 0.0,
        "var_95_pct": var_95,
        "cvar_95_pct": cvar_95,
        "max_drawdown_pct": max_drawdown_pct,
        "guardrail_breached": guardrail_breached,
        "de_risk_factor": de_risk_factor,
    }

    return {
        "portfolio_df": base_df,
        "sector_df": sector_df,
        "corr_df": corr_df,
        "summary": summary,
    }


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

        components = _continuous_signal_components(
            rsi_value=rsi_value,
            macd_value=macd_value,
            macd_signal_value=macd_signal_value,
            close_value=close_value,
            sma_20_value=sma_20_value,
            sma_50_value=sma_50_value,
        )
        momentum_component = _tanh_score(momentum_5 * 100.0, 3.5)
        volatility_ratio = atr / close_value if close_value > 0 else 0.0
        volatility_penalty = _clamp((volatility_ratio / 0.03) - 1.0, 0.0, 1.0)

        score = (
            0.26 * components["rsi_component"]
            + 0.24 * components["macd_component"]
            + 0.20 * components["trend20_component"]
            + 0.16 * components["trend50_component"]
            + 0.14 * momentum_component
            - 0.05 * volatility_penalty
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
                    <div class="subtle-kicker">Market Intelligence Workspace</div>
                    <p class="hero-title"><span class="hero-accent">NSE Signal Studio</span></p>
                    <p class="hero-sub">A modern trading cockpit for EOD signals, derivatives context, portfolio construction, intraday intelligence, and backtesting.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Controls")
    selected_name = st.sidebar.selectbox(
        "Select NIFTY 50 stock",
        list(NIFTY_50_STOCKS.keys()),
        index=0,
    )
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

    signal_tab, pulse_tab, verdict_tab, portfolio_tab, intraday_tab, backtest_tab, nifty_analysis_tab, derivatives_quant_tab = st.tabs(
        [
            "Signal Dashboard",
            "NIFTY 50 Pulse",
            "Verdict",
            "Portfolio Lab",
            "Intraday Intelligence",
            "Backtest",
            "NIFTY 50 Analysis",
            "Derivatives + Quant Lab",
        ]
    )

    with signal_tab:
        eod_df = get_eod_cached(selected_ticker, days=days)
        if eod_df.empty:
            st.error("No EOD data available for selected ticker.")
            fallback_ticker = "RELIANCE.NS"
            eod_df = get_eod_cached(fallback_ticker, days=days)
            if eod_df.empty:
                st.warning("Fallback data is also unavailable. Signal tab is limited right now, but other tabs continue to work.")
                eod_df = pd.DataFrame(
                    {
                        "Open": [0.0],
                        "High": [0.0],
                        "Low": [0.0],
                        "Close": [0.0],
                        "Volume": [0.0],
                        "SMA_20": [0.0],
                        "SMA_50": [0.0],
                        "EMA_20": [0.0],
                        "RSI_14": [50.0],
                        "MACD": [0.0],
                        "MACD_SIGNAL": [0.0],
                        "MACD_HIST": [0.0],
                        "DAILY_RETURN_PCT": [0.0],
                        "DATA_SOURCE": ["unavailable"],
                        "IS_PROVISIONAL": [True],
                    },
                    index=[pd.Timestamp.utcnow().tz_localize(None)],
                )

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

        chart_series_options = ["Close", "SMA_20", "EMA_20", "SMA_50"]
        selected_chart_series = st.multiselect(
            "Visible chart series",
            options=chart_series_options,
            default=chart_series_options,
            help="Toggle individual lines on the price chart.",
        )
        _render_price_structure_chart(
            eod_df,
            selected_name,
            selected_ticker,
            selected_chart_series,
        )

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

    with portfolio_tab:
        st.subheader("Portfolio Lab")
        st.caption("Build multi-position portfolios from verdict candidates with correlation-aware sizing and risk guardrails.")

        pcol1, pcol2, pcol3 = st.columns(3)
        portfolio_policy = pcol1.selectbox("Portfolio policy", list(VERDICT_POLICIES.keys()), index=1, key="portfolio_policy")
        portfolio_capital = pcol2.number_input("Portfolio capital", min_value=50000.0, value=500000.0, step=10000.0, key="portfolio_capital")
        portfolio_positions = int(pcol3.slider("Max positions", min_value=3, max_value=12, value=6, step=1, key="portfolio_positions"))

        pcol4, pcol5, pcol6 = st.columns(3)
        sector_cap_pct = float(pcol4.slider("Max sector exposure (%)", min_value=20, max_value=70, value=35, step=5, key="portfolio_sector_cap"))
        corr_lookback = int(pcol5.slider("Correlation lookback days", min_value=45, max_value=252, value=126, step=9, key="portfolio_corr_lookback"))
        dd_guardrail = float(pcol6.slider("Drawdown guardrail (%)", min_value=4.0, max_value=20.0, value=10.0, step=0.5, key="portfolio_dd_guardrail"))

        if st.button("Construct Portfolio", key="run_portfolio_lab"):
            with st.spinner("Constructing portfolio from verdict candidates..."):
                verdict_pool = build_verdict_candidates(
                    days=days,
                    finalized_only=finalized_only,
                    policy_name=portfolio_policy,
                )
                verdict_pool = apply_position_sizing(
                    verdict_df=verdict_pool,
                    account_capital=float(portfolio_capital),
                    risk_per_trade_pct=1.0,
                    max_trades=max(portfolio_positions * 2, portfolio_positions),
                )

                portfolio_bundle = build_portfolio_from_verdict(
                    verdict_df=verdict_pool,
                    account_capital=float(portfolio_capital),
                    max_positions=portfolio_positions,
                    max_sector_exposure_pct=sector_cap_pct,
                    lookback_days=corr_lookback,
                    drawdown_guardrail_pct=dd_guardrail,
                    finalized_only=finalized_only,
                )
                st.session_state["portfolio_bundle"] = portfolio_bundle

        portfolio_bundle = st.session_state.get("portfolio_bundle")
        if isinstance(portfolio_bundle, dict) and portfolio_bundle:
            summary = portfolio_bundle.get("summary", {})
            portfolio_df = portfolio_bundle.get("portfolio_df", pd.DataFrame())
            sector_df = portfolio_bundle.get("sector_df", pd.DataFrame())
            corr_df = portfolio_bundle.get("corr_df", pd.DataFrame())

            metric_cols = st.columns(6)
            metric_cols[0].metric("Positions", f"{int(summary.get('positions', 0))}")
            metric_cols[1].metric("Capital Used", f"{float(summary.get('capital_used', 0.0)):.2f}")
            metric_cols[2].metric("Portfolio R:R", f"{float(summary.get('portfolio_rr', 0.0)):.2f}")
            metric_cols[3].metric("VaR 95%", f"{float(summary.get('var_95_pct', 0.0)):.2f}%")
            metric_cols[4].metric("CVaR 95%", f"{float(summary.get('cvar_95_pct', 0.0)):.2f}%")
            metric_cols[5].metric("Max Drawdown", f"{float(summary.get('max_drawdown_pct', 0.0)):.2f}%")

            if bool(summary.get("guardrail_breached", False)):
                st.warning(
                    f"Drawdown guardrail breached. Suggested de-risk factor: {float(summary.get('de_risk_factor', 1.0)):.2f}x"
                )
            else:
                st.success("Drawdown guardrail check passed for current portfolio mix.")

            if isinstance(portfolio_df, pd.DataFrame) and not portfolio_df.empty:
                st.markdown("##### Portfolio Construction")
                view_cols = [
                    "company",
                    "ticker",
                    "sector",
                    "side",
                    "score",
                    "rr_ratio",
                    "avg_corr",
                    "alloc_weight",
                    "capital_alloc",
                    "qty_alloc",
                    "entry",
                    "target",
                    "stop_loss",
                    "expected_profit",
                    "expected_loss",
                ]
                available_cols = [col for col in view_cols if col in portfolio_df.columns]
                st.dataframe(
                    portfolio_df[available_cols].style.format(
                        {
                            "score": "{:.3f}",
                            "rr_ratio": "{:.2f}",
                            "avg_corr": "{:.2f}",
                            "alloc_weight": "{:.2%}",
                            "capital_alloc": "{:.2f}",
                            "entry": "{:.2f}",
                            "target": "{:.2f}",
                            "stop_loss": "{:.2f}",
                            "expected_profit": "{:.2f}",
                            "expected_loss": "{:.2f}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            s1, s2 = st.columns(2)
            with s1:
                st.markdown("##### Sector Exposure")
                if isinstance(sector_df, pd.DataFrame) and not sector_df.empty:
                    st.dataframe(
                        sector_df[["sector", "weight_pct"]].style.format({"weight_pct": "{:.2f}%"}),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("Sector exposure is unavailable.")

            with s2:
                st.markdown("##### Correlation Matrix")
                if isinstance(corr_df, pd.DataFrame) and not corr_df.empty:
                    st.dataframe(corr_df.style.format("{:.2f}"), use_container_width=True)
                else:
                    st.info("Correlation matrix not available for selected set.")
        else:
            st.info("Click 'Construct Portfolio' to generate a correlation-aware multi-position allocation.")

    with intraday_tab:
        st.subheader("Intraday Intelligence")
        st.caption("5m/15m session analytics with opening range, gap playbook, pivots, and breakout failure stats.")

        ic1, ic2, ic3 = st.columns(3)
        intraday_name = ic1.selectbox("Intraday symbol", list(NIFTY_50_STOCKS.keys()), index=list(NIFTY_50_STOCKS.keys()).index(selected_name), key="intraday_symbol")
        intraday_interval = ic2.selectbox("Timeframe", options=["5m", "15m"], index=0, key="intraday_interval")
        intraday_period_days = int(ic3.slider("Intraday lookback (days)", min_value=1, max_value=10, value=5, step=1, key="intraday_days"))

        opening_range_mins = int(
            st.selectbox("Opening range window", options=[15, 30, 45, 60], index=1, key="intraday_or_window")
        )

        intraday_ticker = NIFTY_50_STOCKS[intraday_name]
        with st.spinner("Fetching intraday bars..."):
            intraday_df = fetch_intraday_data(intraday_ticker, interval=intraday_interval, period_days=intraday_period_days)

        if intraday_df.empty:
            st.warning("Intraday data unavailable for this ticker/timeframe right now.")
        else:
            or_stats = _compute_opening_range_stats(intraday_df, opening_range_mins)
            touch_df, pivot_stats = _compute_intraday_pivot_reactions(intraday_df)
            gap_playbook = _build_preopen_gap_playbook(intraday_ticker, intraday_df)

            latest_price = float(intraday_df["Close"].iloc[-1]) if "Close" in intraday_df else 0.0
            session_open = float(intraday_df[intraday_df.index.date == intraday_df.index[-1].date()]["Open"].iloc[0]) if "Open" in intraday_df else latest_price
            intraday_move = ((latest_price - session_open) / session_open) * 100.0 if session_open > 0 else 0.0

            mcols = st.columns(6)
            mcols[0].metric("Live Price", f"{latest_price:.2f}")
            mcols[1].metric("Session Move", f"{intraday_move:.2f}%")
            mcols[2].metric("OR High", f"{float(or_stats.get('opening_high', 0.0)):.2f}")
            mcols[3].metric("OR Low", f"{float(or_stats.get('opening_low', 0.0)):.2f}")
            mcols[4].metric("Breakout Bias", str(or_stats.get("breakout_bias", "N/A")))
            mcols[5].metric("Pivot Touches", f"{float(pivot_stats.get('pivot_touch_total', 0.0)):.0f}")

            st.markdown("##### Pre-open Gap Playbook + Opening Auction Signal")
            if gap_playbook:
                g1, g2, g3, g4 = st.columns(4)
                g1.metric("Gap %", f"{float(gap_playbook.get('gap_pct', 0.0)):.2f}%")
                g2.metric("Auction Return", f"{float(gap_playbook.get('auction_return_pct', 0.0)):.2f}%")
                g3.metric("Auction Volume Ratio", f"{float(gap_playbook.get('auction_volume_ratio', 1.0)):.2f}x")
                g4.metric("Auction Signal", str(gap_playbook.get("auction_signal", "N/A")))
                st.info(f"Playbook: {gap_playbook.get('playbook', 'N/A')}")
            else:
                st.info("Gap playbook data unavailable for this session.")

            st.markdown("##### Live Session Dashboard")
            chart_df = intraday_df.copy().reset_index().rename(columns={"index": "Timestamp"})
            if "Datetime" in chart_df.columns:
                chart_df = chart_df.rename(columns={"Datetime": "Timestamp"})
            if "Timestamp" not in chart_df.columns:
                chart_df["Timestamp"] = pd.to_datetime(chart_df.index)

            base = alt.Chart(chart_df).encode(x=alt.X("Timestamp:T", title=None))
            price_line = base.mark_line(color="#74c0fc", strokeWidth=2).encode(
                y=alt.Y("Close:Q", title="Price")
            )

            overlays = []
            if or_stats:
                overlays.append(
                    base.mark_rule(color="#f97316", strokeDash=[6, 4]).encode(y=alt.datum(float(or_stats.get("opening_high", 0.0))))
                )
                overlays.append(
                    base.mark_rule(color="#ef4444", strokeDash=[6, 4]).encode(y=alt.datum(float(or_stats.get("opening_low", 0.0))))
                )

            if not touch_df.empty:
                for level_row in touch_df.itertuples(index=False):
                    color = "#22c55e" if str(level_row.Level).startswith("S") else "#a78bfa" if str(level_row.Level).startswith("R") else "#facc15"
                    overlays.append(
                        base.mark_rule(color=color, opacity=0.35).encode(y=alt.datum(float(level_row.Price)))
                    )

            chart = price_line
            for layer in overlays:
                chart = chart + layer
            st.altair_chart(chart.properties(height=320).interactive(), use_container_width=True)

            stat_cols = st.columns(2)
            stat_cols[0].metric("Failed R1 Breakouts", f"{int(pivot_stats.get('failed_r1_breaks', 0))}")
            stat_cols[1].metric("Failed S1 Breakdowns", f"{int(pivot_stats.get('failed_s1_breaks', 0))}")

            st.markdown("##### Intraday Pivot Reactions")
            if touch_df.empty:
                st.info("Pivot reaction stats are not available yet.")
            else:
                st.dataframe(
                    touch_df.style.format({"Price": "{:.2f}", "Touches": "{:.0f}"}),
                    use_container_width=True,
                    hide_index=True,
                )

    with backtest_tab:
        st.subheader("Recommendation Backtest")
        log_df = load_recommendation_log()
        if log_df.empty:
            st.info(
                "No recommendation snapshots found yet. Log a few snapshots first.")
        else:
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
            else:
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

    with nifty_analysis_tab:
        st.subheader("NIFTY 50 Index Analysis")
        st.caption("Live index data, technical indicators, support/resistance, and market breadth.")

        index_days = st.slider("Index lookback days", min_value=30, max_value=180, value=60, key="index_days")

        with st.spinner("Fetching NIFTY 50 index data..."):
            index_df = fetch_nifty50_index_data(days=index_days)

        if index_df.empty:
            st.error("Unable to fetch NIFTY 50 index data. Please try again later.")
        else:
            latest_idx = index_df.iloc[-1]
            close_value = _safe_float(latest_idx.get("Close"), 0.0)
            prev_close = _safe_float(index_df.iloc[-2].get("Close"), close_value) if len(index_df) >= 2 else close_value
            day_change_pct = ((close_value - prev_close) / prev_close) * 100 if prev_close > 0 else 0.0

            rsi_value = _safe_float(latest_idx.get("RSI_14"), 50.0)
            macd_value = _safe_float(latest_idx.get("MACD"), 0.0)
            macd_signal_value = _safe_float(latest_idx.get("MACD_SIGNAL"), 0.0)
            sma_20_value = _safe_float(latest_idx.get("SMA_20"), close_value)
            sma_50_value = _safe_float(latest_idx.get("SMA_50"), close_value)
            ema_20_value = _safe_float(latest_idx.get("EMA_20"), close_value)

            latest_date = pd.to_datetime(index_df.index[-1]).date()

            st.markdown(f"**Last Updated:** {latest_date}")
            st.markdown("---")

            metric_cols = st.columns(6)
            metric_cols[0].metric("NIFTY 50", f"{close_value:.2f}")
            metric_cols[1].metric("1D Change", f"{day_change_pct:.2f}%")
            metric_cols[2].metric("RSI (14)", f"{rsi_value:.2f}")
            metric_cols[3].metric("MACD", f"{macd_value:.2f}")
            metric_cols[4].metric("MACD Signal", f"{macd_signal_value:.2f}")
            metric_cols[5].metric("MACD Hist", f"{_safe_float(latest_idx.get('MACD_HIST'), 0.0):.2f}")

            st.markdown("#### Moving Averages")
            ma_cols = st.columns(4)
            ma_cols[0].metric("SMA (20)", f"{sma_20_value:.2f}")
            ma_cols[1].metric("SMA (50)", f"{sma_50_value:.2f}")
            ma_cols[2].metric("EMA (20)", f"{ema_20_value:.2f}")
            ma_cols[3].metric("Distance to SMA20", f"{((close_value - sma_20_value) / sma_20_value * 100):.2f}%")

            support_df, resistance_df = _calculate_support_resistance_levels(
                index_df,
                close_value,
                lookback_days=min(len(index_df), 120),
                pivot_window=3,
                max_levels=3,
            )

            st.markdown("#### Support & Resistance")
            if support_df.empty and resistance_df.empty:
                st.info("Not enough data to calculate support and resistance levels.")
            else:
                sr_cols = st.columns(4)
                top_support = support_df.iloc[0] if not support_df.empty else None
                top_resistance = resistance_df.iloc[0] if not resistance_df.empty else None

                sr_cols[0].metric(
                    "Nearest Support",
                    f"{float(top_support['Level']):.2f}" if top_support is not None else "N/A",
                    f"{float(top_support['Distance to Close (%)']):.2f}% below" if top_support is not None else "",
                )
                sr_cols[1].metric(
                    "Support Strength",
                    f"{float(top_support['Strength']):.2f}" if top_support is not None else "N/A",
                    f"{int(top_support['Touches'])} touches" if top_support is not None else "",
                )
                sr_cols[2].metric(
                    "Nearest Resistance",
                    f"{float(top_resistance['Level']):.2f}" if top_resistance is not None else "N/A",
                    f"{float(top_resistance['Distance to Close (%)']):.2f}% above" if top_resistance is not None else "",
                )
                sr_cols[3].metric(
                    "Resistance Strength",
                    f"{float(top_resistance['Strength']):.2f}" if top_resistance is not None else "N/A",
                    f"{int(top_resistance['Touches'])} touches" if top_resistance is not None else "",
                )

                all_levels = []
                if not support_df.empty:
                    all_levels.append(support_df)
                if not resistance_df.empty:
                    all_levels.append(resistance_df)

                if all_levels:
                    levels_combined = pd.concat(all_levels, ignore_index=True)
                    levels_combined = levels_combined.sort_values(["Level Type", "Level"], ascending=[True, False])
                    st.dataframe(levels_combined.style.format({
                        "Level": "{:.2f}",
                        "Distance to Close (%)": "{:.2f}",
                        "Strength": "{:.2f}",
                    }), use_container_width=True, hide_index=True)

            next_session_levels = _calculate_next_session_pivots(index_df)
            st.markdown("#### Next Trading Session Levels (Pivot-Based)")
            st.caption("Derived from the latest completed NIFTY 50 OHLC bar for the next session's intraday reference.")
            if next_session_levels.empty:
                st.info("Next-session pivot levels are unavailable due to missing OHLC data.")
            else:
                pivot_row = next_session_levels[next_session_levels["Level"] == "Pivot"]
                r1_row = next_session_levels[next_session_levels["Level"] == "R1"]
                s1_row = next_session_levels[next_session_levels["Level"] == "S1"]

                pivot_value = float(pivot_row.iloc[0]["Price"]) if not pivot_row.empty else 0.0
                r1_value = float(r1_row.iloc[0]["Price"]) if not r1_row.empty else 0.0
                s1_value = float(s1_row.iloc[0]["Price"]) if not s1_row.empty else 0.0

                next_cols = st.columns(3)
                next_cols[0].metric("Pivot", f"{pivot_value:.2f}")
                next_cols[1].metric("Nearest Resistance (R1)", f"{r1_value:.2f}")
                next_cols[2].metric("Nearest Support (S1)", f"{s1_value:.2f}")

                ordered_levels = ["R3", "R2", "R1", "Pivot", "S1", "S2", "S3"]
                next_session_levels["order_key"] = next_session_levels["Level"].map(
                    {name: idx for idx, name in enumerate(ordered_levels)}
                )
                next_session_levels = next_session_levels.sort_values("order_key").drop(columns=["order_key"])

                st.dataframe(
                    next_session_levels.style.format(
                        {
                            "Price": "{:.2f}",
                            "Distance to Close (%)": "{:.2f}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            breadth = calculate_market_breadth(index_df)

            st.markdown("#### Market Breadth Indicators")
            st.caption("Based on index historical data and momentum")
            breadth_cols = st.columns(4)
            breadth_cols[0].metric("% Above SMA20", breadth.get("percent_above_sma20", "N/A"))
            breadth_cols[1].metric("% Above SMA50", breadth.get("percent_above_sma50", "N/A"))
            breadth_cols[2].metric("Trend Strength", breadth.get("trend_strength", "N/A"))
            breadth_cols[3].metric("5-Day Momentum", breadth.get("5day_momentum", "N/A"))

            st.markdown("#### Historical Price Chart")
            chart_series = st.multiselect(
                "Chart series to display",
                options=["Close", "SMA_20", "SMA_50", "EMA_20"],
                default=["Close", "SMA_20", "SMA_50"],
                key="nifty_chart_series"
            )

            if chart_series:
                chart_df = index_df[chart_series].copy().dropna()
                if not chart_df.empty:
                    chart_df_reset = chart_df.reset_index().rename(columns={"index": "Date"})

                    series_colors = {
                        "Close": "#74c0fc",
                        "SMA_20": "#f97316",
                        "SMA_50": "#ef4444",
                        "EMA_20": "#38bdf8",
                    }

                    long_chart_df = chart_df_reset.melt(
                        id_vars=["Date"],
                        value_vars=chart_series,
                        var_name="Series",
                        value_name="Price",
                    ).dropna(subset=["Price"])

                    if not long_chart_df.empty:
                        y_min = float(long_chart_df["Price"].min())
                        y_max = float(long_chart_df["Price"].max())
                        padding = max((y_max - y_min) * 0.05, y_max * 0.002)

                        color_scale = alt.Scale(
                            domain=chart_series,
                            range=[series_colors[s] for s in chart_series if s in series_colors],
                        )

                        chart = (
                            alt.Chart(long_chart_df)
                            .mark_line(strokeWidth=2)
                            .encode(
                                x=alt.X("Date:T", title=None),
                                y=alt.Y("Price:Q", title="Price", scale=alt.Scale(domain=[max(0, y_min - padding), y_max + padding])),
                                color=alt.Color("Series:N", scale=color_scale, legend=alt.Legend(orient="top")),
                                tooltip=[
                                    alt.Tooltip("Date:T", title="Date"),
                                    alt.Tooltip("Series:N", title="Series"),
                                    alt.Tooltip("Price:Q", title="Price", format=",.2f"),
                                ],
                            )
                            .properties(height=320)
                            .interactive()
                        )
                        st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("Insufficient data for chart display.")

            st.markdown("#### Technical Summary")
            col1, col2 = st.columns(2)

            with col1:
                trend_label = "Bullish" if close_value > sma_20_value and close_value > sma_50_value else "Bearish" if close_value < sma_20_value and close_value < sma_50_value else "Neutral"
                rsi_label = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
                macd_signal = "Bullish" if macd_value > macd_signal_value else "Bearish"

                st.markdown(
                    f"""
                    **Trend:** {trend_label}
                    - Price above SMA20: {'Yes' if close_value > sma_20_value else 'No'}
                    - Price above SMA50: {'Yes' if close_value > sma_50_value else 'No'}

                    **Momentum:**
                    - RSI (14): {rsi_value:.2f} ({rsi_label})
                    - MACD: {macd_signal}
                    - MACD Histogram: {'Positive' if macd_value > macd_signal_value else 'Negative'}
                    """
                )

            with col2:
                volatility_20 = float(index_df["DAILY_RETURN_PCT"].tail(20).std()) if "DAILY_RETURN_PCT" in index_df else 0.0
                avg_volume = int(index_df["Volume"].tail(20).mean()) if "Volume" in index_df else 0

                st.markdown(
                    f"""
                    **Volatility:**
                    - 20D Std Dev: {volatility_20:.2f}%

                    **Volume:**
                    - 20D Avg Volume: {avg_volume:,.0f}

                    **Data:**
                    - Lookback: {index_days} days
                    - Total bars: {len(index_df)}
                    """
                )

            st.markdown("---")
            st.caption("Price-action analytics shown above are complemented by the Derivatives + Quant Lab tab for option-chain and probabilistic views.")

    with derivatives_quant_tab:
        st.subheader("Derivatives + Advanced Quant Lab")
        st.caption("NIFTY option-chain analytics plus feature-store driven probabilistic ranking.")

        st.markdown("#### Options and Derivatives Layer")
        with st.spinner("Fetching NSE option-chain snapshot..."):
            options_snapshot = fetch_nifty_option_chain("NIFTY")

        if not options_snapshot:
            st.warning("NSE option-chain snapshot is currently unavailable. Retry in a few seconds.")
        else:
            options_metrics, magnets_df = _compute_derivatives_insights(options_snapshot)
            iv_percentile = _update_iv_history_and_percentile(
                expiry=str(options_metrics.get("expiry", "N/A")),
                atm_iv=float(options_metrics.get("atm_iv", 0.0)),
            )

            top_row = st.columns(6)
            top_row[0].metric("NIFTY Spot", f"{float(options_metrics.get('underlying', 0.0)):.2f}")
            top_row[1].metric("Put/Call OI Ratio", f"{float(options_metrics.get('pcr', 0.0)):.2f}")
            top_row[2].metric("Max Pain", f"{float(options_metrics.get('max_pain', 0.0)):.0f}")
            top_row[3].metric("Gamma Wall", f"{float(options_metrics.get('gamma_wall', 0.0)):.0f}")
            top_row[4].metric("ATM IV", f"{float(options_metrics.get('atm_iv', 0.0)):.2f}")
            top_row[5].metric("IV Percentile", f"{iv_percentile:.1f}%" if iv_percentile is not None else "N/A")

            exp_cols = st.columns(2)
            exp_cols[0].metric("Nearest Expiry", str(options_metrics.get("expiry", "N/A")))
            dte = _days_to_expiry(str(options_metrics.get("expiry", "N/A")))
            exp_cols[1].metric("Days To Expiry", str(dte) if dte is not None else "N/A")

            st.markdown("##### Key Strike Magnets (OI + Gamma Concentration)")
            if magnets_df.empty:
                st.info("No strike-level magnet data available for this snapshot.")
            else:
                st.dataframe(
                    magnets_df.style.format(
                        {
                            "strike": "{:.0f}",
                            "total_oi": "{:,.0f}",
                            "gamma_exposure": "{:,.2f}",
                            "distance_to_spot_pct": "{:.2f}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        st.markdown("#### Event-Day Mode")
        inferred_dte = _days_to_expiry(str(options_metrics.get("expiry", "N/A"))) if options_snapshot else None
        default_template = "Expiry Day" if inferred_dte is not None and inferred_dte <= 1 else "Normal Session"
        event_mode = st.toggle(
            "Enable event-day risk template",
            value=False,
            help="Apply stricter risk parameters for expiry and macro-event sessions.",
            key="event_day_mode_toggle",
        )
        selected_template = st.selectbox(
            "Risk template",
            options=list(EVENT_DAY_RISK_TEMPLATES.keys()),
            index=list(EVENT_DAY_RISK_TEMPLATES.keys()).index(default_template),
            key="event_day_risk_template",
        )
        template = EVENT_DAY_RISK_TEMPLATES[selected_template if event_mode else "Normal Session"]
        risk_cols = st.columns(4)
        risk_cols[0].metric("Max Risk / Trade", f"{float(template['max_risk_per_trade_pct']):.2f}%")
        risk_cols[1].metric("Max Open Positions", f"{int(template['max_open_positions'])}")
        risk_cols[2].metric("Stop Buffer (ATR)", f"{float(template['stop_buffer_atr']):.2f}")
        risk_cols[3].metric("Target Buffer (ATR)", f"{float(template['target_buffer_atr']):.2f}")

        st.markdown("#### Advanced Analytics Layer")
        st.caption("Feature store includes technical, sentiment, and lightweight fundamental factors; outputs blend rule-engine and ML rank.")

        if st.button("Build Feature Store + Run Meta Model", key="run_advanced_analytics"):
            with st.spinner("Building feature store and running probabilistic meta-model..."):
                feature_df = build_feature_store_snapshot(
                    days=days,
                    finalized_only=finalized_only,
                    use_live_news=use_live_news,
                    max_news_items=news_items,
                )

                if feature_df.empty:
                    st.warning("Feature store could not be built. Check connectivity or increase lookback.")
                else:
                    store_path = _persist_feature_store_snapshot(feature_df)
                    meta_df = run_meta_model_blend(feature_df)
                    st.session_state["latest_feature_store_df"] = feature_df
                    st.session_state["latest_meta_df"] = meta_df
                    st.success(f"Feature store updated: {store_path}")

        feature_store_df = st.session_state.get("latest_feature_store_df")
        meta_rank_df = st.session_state.get("latest_meta_df")

        if isinstance(meta_rank_df, pd.DataFrame) and not meta_rank_df.empty:
            st.markdown("##### Meta-Model Ranked Signals")
            display_cols = [
                "company",
                "ticker",
                "rule_recommendation",
                "rule_score",
                "ml_expected_return_pct",
                "meta_score",
                "meta_recommendation",
                "prob_up",
                "prob_down",
                "prob_flat",
                "expected_return_pct",
            ]
            available = [col for col in display_cols if col in meta_rank_df.columns]
            st.dataframe(
                meta_rank_df[available].head(20).style.format(
                    {
                        "rule_score": "{:.3f}",
                        "ml_expected_return_pct": "{:.3f}",
                        "meta_score": "{:.3f}",
                        "prob_up": "{:.2%}",
                        "prob_down": "{:.2%}",
                        "prob_flat": "{:.2%}",
                        "expected_return_pct": "{:.2f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download Meta-Model Rankings CSV",
                data=meta_rank_df.to_csv(index=False),
                file_name=f"meta_rankings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_meta_rankings",
            )

        if isinstance(feature_store_df, pd.DataFrame) and not feature_store_df.empty:
            st.markdown("##### Latest Feature Store Snapshot")
            st.dataframe(feature_store_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
