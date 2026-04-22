"""Data and sentiment utilities for NSE pre/post-market analysis."""

from __future__ import annotations

import io
import zipfile
from typing import Callable, Dict, List

import feedparser
import pandas as pd
import requests
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    import pandas_ta as ta
except ImportError:
    ta = None


MOCK_NEWS_BY_TICKER: Dict[str, List[str]] = {
    "RELIANCE.NS": [
        "Reliance expands renewable energy investments with new solar project",
        "Brokerages maintain positive outlook on Reliance retail growth",
        "Reliance telecom unit reports steady subscriber additions",
        "Crude volatility keeps pressure on energy sector margins",
        "Institutional investors increase exposure to Reliance before earnings",
    ],
    "TCS.NS": [
        "TCS secures multi-year digital transformation contract in Europe",
        "Management signals cautious optimism on FY guidance",
        "Rupee weakness seen supporting large IT exporters",
        "Analysts flag slower discretionary spending in key US clients",
        "TCS continues hiring for AI and cloud delivery roles",
    ],
    "HDFCBANK.NS": [
        "HDFC Bank credit growth remains resilient in latest update",
        "Broker report highlights improving net interest margins",
        "Banking sector may face near-term deposit cost pressure",
        "HDFC Bank expands branch network in tier-2 cities",
        "Foreign funds continue accumulation in private banks",
    ],
    "INFY.NS": [
        "Infosys signs strategic partnership focused on enterprise AI",
        "Street watches deal pipeline commentary for demand trends",
        "Infosys margin discipline praised in recent analyst note",
        "Concerns remain around weak technology budgets in some verticals",
        "Large deal wins support medium-term revenue visibility",
    ],
    "ICICIBANK.NS": [
        "ICICI Bank posts healthy loan growth across retail segments",
        "Analysts expect stable asset quality for private lenders",
        "Treasury gains may remain limited amid bond yield fluctuations",
        "ICICI Bank digital initiatives drive customer onboarding",
        "Market participants positive on private banking sector momentum",
    ],
}

RSS_FEEDS = [
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://www.moneycontrol.com/rss/marketreports.xml",
    "https://www.moneycontrol.com/rss/business.xml",
]

TICKER_KEYWORDS: Dict[str, List[str]] = {
    "RELIANCE.NS": [
        "reliance",
        "reliance industries",
        "ril",
        "jioplatforms",
        "jio",
    ],
    "TCS.NS": [
        "tcs",
        "tata consultancy",
        "tata consultancy services",
        "tata group it",
    ],
    "HDFCBANK.NS": [
        "hdfc bank",
        "hdfcbank",
        "housing development finance corporation bank",
    ],
    "INFY.NS": [
        "infosys",
        "infy",
        "infosys ltd",
        "infosys limited",
    ],
    "ICICIBANK.NS": [
        "icici bank",
        "icicibank",
        "industrial credit and investment corporation of india",
    ],
}

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/csv,application/zip,application/octet-stream,text/plain,*/*",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
}

_BHAVCOPY_CACHE: Dict[str, pd.DataFrame] = {}


def _normalize_text(text: str) -> str:
    """Lowercase and normalize separators for simple keyword matching."""
    cleaned = "".join(ch if ch.isalnum() else " " for ch in text.lower())
    return " ".join(cleaned.split())


def _build_ticker_keywords(ticker: str) -> List[str]:
    """Build de-duplicated keyword aliases for ticker-level news filtering."""
    base_symbol = ticker.replace(".NS", "").lower()
    dynamic_aliases = [base_symbol, f"{base_symbol} ns"]
    configured = TICKER_KEYWORDS.get(ticker, [])
    merged = configured + dynamic_aliases

    deduped: List[str] = []
    seen = set()
    for keyword in merged:
        normalized = _normalize_text(keyword)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def fetch_eod_data(ticker: str, days: int = 30) -> pd.DataFrame:
    """Fetch EOD OHLCV data and calculate RSI and MACD.

    Args:
        ticker: NSE ticker in yfinance format, e.g., "RELIANCE.NS".
        days: Number of trailing trading days to return.

    Returns:
        DataFrame with OHLCV and indicator columns.
    """
    lookback_days = max(120, days + 60)
    ist_now = pd.Timestamp.now(tz="Asia/Kolkata")
    end_date = (ist_now + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (ist_now - pd.Timedelta(days=lookback_days * 2)
                  ).strftime("%Y-%m-%d")

    # Method 1: explicit date-window download.
    range_df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False,
    )

    # Method 2: rolling period download.
    period_df = yf.download(
        ticker,
        period=f"{lookback_days}d",
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False,
    )

    # Method 3: ticker.history often updates first for some exchanges.
    history_df = yf.Ticker(ticker).history(
        period="10d", interval="1d", auto_adjust=False)

    frames = []
    for raw_df in [range_df, period_df, history_df]:
        normalized = _normalize_price_frame(raw_df)
        if not normalized.empty:
            frames.append(normalized)

    if not frames:
        return pd.DataFrame()

    price_df = pd.concat(frames, axis=0).sort_index()
    price_df = price_df[~price_df.index.duplicated(keep="last")]
    price_df["DATA_SOURCE"] = "yfinance"
    price_df["IS_PROVISIONAL"] = False

    # Prefer official NSE EOD rows where available.
    price_df = _apply_nse_official_patch(price_df, ticker)

    # If latest close is still missing, patch with quote endpoint as provisional.
    price_df = _patch_latest_quote_row(price_df, ticker)

    price_df = price_df.dropna(subset=["Close"]).copy()
    price_df["SMA_20"] = price_df["Close"].rolling(20).mean()
    price_df["SMA_50"] = price_df["Close"].rolling(50).mean()
    price_df["EMA_20"] = price_df["Close"].ewm(span=20, adjust=False).mean()
    price_df["DAILY_RETURN_PCT"] = price_df["Close"].pct_change() * 100

    if ta is not None:
        price_df["RSI_14"] = ta.rsi(price_df["Close"], length=14)
        macd_df = ta.macd(price_df["Close"], fast=12, slow=26, signal=9)

        if macd_df is not None and not macd_df.empty:
            price_df["MACD"] = macd_df["MACD_12_26_9"]
            price_df["MACD_SIGNAL"] = macd_df["MACDs_12_26_9"]
            price_df["MACD_HIST"] = macd_df["MACDh_12_26_9"]
        else:
            price_df["MACD"] = pd.NA
            price_df["MACD_SIGNAL"] = pd.NA
            price_df["MACD_HIST"] = pd.NA
    else:
        price_df["RSI_14"] = _compute_rsi(price_df["Close"], window=14)
        macd, macd_signal, macd_hist = _compute_macd(
            price_df["Close"], fast=12, slow=26, signal=9)
        price_df["MACD"] = macd
        price_df["MACD_SIGNAL"] = macd_signal
        price_df["MACD_HIST"] = macd_hist

    return price_df.tail(days)


def _normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance outputs to a consistent OHLCV frame."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    existing = [col for col in required_cols if col in out.columns]
    if not existing:
        return pd.DataFrame()

    out = out[existing]
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out


def _patch_latest_quote_row(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Patch latest row when historical Close is missing but quote endpoint has data."""
    if df.empty or "Close" not in df.columns:
        return df

    out = df.copy()
    latest_idx = out.index.max()
    latest_close = out.at[latest_idx, "Close"]
    if pd.notna(latest_close):
        return out

    fast = getattr(yf.Ticker(ticker), "fast_info", None)
    if fast is None or not hasattr(fast, "get"):
        return out

    last_price = fast.get("lastPrice")
    if last_price is None or pd.isna(last_price):
        return out

    out.at[latest_idx, "Close"] = float(last_price)
    out.at[latest_idx, "DATA_SOURCE"] = "yfinance_quote_patch"
    out.at[latest_idx, "IS_PROVISIONAL"] = True

    day_high = fast.get("dayHigh")
    day_low = fast.get("dayLow")
    last_volume = fast.get("lastVolume")

    if "High" in out.columns and pd.isna(out.at[latest_idx, "High"]) and day_high is not None:
        out.at[latest_idx, "High"] = float(day_high)
    if "Low" in out.columns and pd.isna(out.at[latest_idx, "Low"]) and day_low is not None:
        out.at[latest_idx, "Low"] = float(day_low)
    if "Open" in out.columns and pd.isna(out.at[latest_idx, "Open"]):
        out.at[latest_idx, "Open"] = float(last_price)
    if "Volume" in out.columns and pd.isna(out.at[latest_idx, "Volume"]) and last_volume is not None:
        out.at[latest_idx, "Volume"] = float(last_volume)

    return out


def _apply_nse_official_patch(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Patch/append the latest official NSE bhavcopy row when available."""
    if df.empty:
        return df

    symbol = ticker.replace(".NS", "")
    out = df.copy()
    ist_today = pd.Timestamp.now(
        tz="Asia/Kolkata").normalize().tz_localize(None)

    for offset in range(0, 8):
        trade_date = ist_today - pd.Timedelta(days=offset)
        bhav_df = _get_nse_bhavcopy(trade_date)
        if bhav_df.empty:
            continue

        row_df = bhav_df[bhav_df["SYMBOL"] == symbol]
        if row_df.empty:
            continue

        row = row_df.iloc[0]
        row_values = {
            "Open": _safe_num(row.get("OPEN")),
            "High": _safe_num(row.get("HIGH")),
            "Low": _safe_num(row.get("LOW")),
            "Close": _safe_num(row.get("CLOSE")),
            "Volume": _safe_num(row.get("TOTTRDQTY")),
            "DATA_SOURCE": "nse_bhavcopy",
            "IS_PROVISIONAL": False,
        }

        idx = pd.Timestamp(trade_date)
        if idx not in out.index:
            out.loc[idx, row_values.keys()] = row_values.values()
        else:
            for key, value in row_values.items():
                out.at[idx, key] = value
        return out.sort_index()

    return out


def _get_nse_bhavcopy(trade_date: pd.Timestamp) -> pd.DataFrame:
    """Fetch daily NSE bhavcopy for a date and cache by date string."""
    key = trade_date.strftime("%Y-%m-%d")
    if key in _BHAVCOPY_CACHE:
        return _BHAVCOPY_CACHE[key]

    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        session.get("https://www.nseindia.com", timeout=6)
    except requests.RequestException:
        pass

    for url in _nse_bhavcopy_urls(trade_date):
        try:
            response = session.get(url, timeout=8)
            if response.status_code != 200:
                continue
        except requests.RequestException:
            continue

        csv_text = _extract_csv_text(response.content)
        if not csv_text:
            continue

        bhav_df = _parse_bhavcopy_csv(csv_text)
        if not bhav_df.empty:
            _BHAVCOPY_CACHE[key] = bhav_df
            return bhav_df

    _BHAVCOPY_CACHE[key] = pd.DataFrame()
    return _BHAVCOPY_CACHE[key]


def _nse_bhavcopy_urls(trade_date: pd.Timestamp) -> List[str]:
    """Return candidate URLs for NSE daily bhavcopy files."""
    ddmmyy = trade_date.strftime("%d%m%y")
    dd_mon_yyyy = trade_date.strftime("%d%b%Y").upper()
    yyyy = trade_date.strftime("%Y")
    mon = trade_date.strftime("%b").upper()

    return [
        f"https://archives.nseindia.com/content/equities/EQ_ISINCODE_{ddmmyy}.csv",
        f"https://archives.nseindia.com/content/historical/EQUITIES/{yyyy}/{mon}/cm{dd_mon_yyyy}bhav.csv.zip",
        f"https://www.nseindia.com/content/historical/EQUITIES/{yyyy}/{mon}/cm{dd_mon_yyyy}bhav.csv.zip",
    ]


def _extract_csv_text(content: bytes) -> str | None:
    """Extract CSV text from raw response bytes (zip or plain csv)."""
    if not content:
        return None

    stripped = content.lstrip()
    if stripped.startswith(b"PK"):
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                names = [name for name in zf.namelist(
                ) if name.lower().endswith(".csv")]
                if not names:
                    return None
                return zf.read(names[0]).decode("utf-8", errors="ignore")
        except zipfile.BadZipFile:
            return None

    text = content.decode("utf-8", errors="ignore")
    if "<!DOCTYPE html" in text[:200].upper():
        return None
    return text


def _parse_bhavcopy_csv(csv_text: str) -> pd.DataFrame:
    """Parse NSE bhavcopy CSV content into a cleaned DataFrame."""
    if not csv_text:
        return pd.DataFrame()

    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()

    expected = {"SYMBOL", "OPEN", "HIGH", "LOW", "CLOSE", "TOTTRDQTY"}
    if not expected.issubset(set(df.columns)):
        return pd.DataFrame()

    df = df.copy()
    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip().str.upper()
    if "SERIES" in df.columns:
        df = df[df["SERIES"].astype(str).str.strip().str.upper() == "EQ"]
    return df


def _safe_num(value: object) -> float | None:
    """Convert values to float safely for row patching."""
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(num):
        return None
    return num


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Compute RSI with Wilder's smoothing using pandas operations."""
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(alpha=1 / window, adjust=False,
                         min_periods=window).mean()
    avg_loss = losses.ewm(alpha=1 / window, adjust=False,
                          min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def _compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD line, signal line, and histogram with EMA."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _fetch_rss_headlines(ticker: str, max_items: int = 5) -> List[str]:
    """Fetch ticker-relevant headlines from configured RSS feeds."""
    keywords = _build_ticker_keywords(ticker)
    matched: List[str] = []
    seen = set()

    for feed_url in RSS_FEEDS:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries:
            title = str(getattr(entry, "title", "")).strip()
            summary = str(getattr(entry, "summary", "")).strip()
            if not title:
                continue
            haystack = _normalize_text(f"{title} {summary}")
            padded_haystack = f" {haystack} "
            if not any(f" {keyword} " in padded_haystack for keyword in keywords):
                continue
            if title in seen:
                continue
            seen.add(title)
            matched.append(title)
            if len(matched) >= max_items:
                return matched

    return matched


def _score_headlines(
    headlines: List[str],
    scorer: Callable[[str], float] | None = None,
) -> pd.DataFrame:
    """Score headline sentiment; scorer can be replaced by an LLM backend later."""
    default_analyzer = SentimentIntensityAnalyzer()
    score_fn = scorer or (
        lambda text: default_analyzer.polarity_scores(text)["compound"])
    records = []

    for headline in headlines:
        raw_score = float(score_fn(headline))
        score = max(min(raw_score, 1.0), -1.0)
        label = "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
        records.append(
            {
                "headline": headline,
                "sentiment_score": round(score, 3),
                "sentiment_label": label,
            }
        )

    return pd.DataFrame(records)


def fetch_news_sentiment(
    ticker: str,
    max_items: int = 5,
    use_live_news: bool = True,
    scorer: Callable[[str], float] | None = None,
) -> pd.DataFrame:
    """Fetch recent headlines and score sentiment from -1 to 1.

    This function is intentionally structured so sentiment can be swapped
    later with another backend such as an LLM API.
    """
    headlines: List[str] = []
    source = "mock"

    if use_live_news:
        headlines = _fetch_rss_headlines(ticker, max_items=max_items)
        if headlines:
            source = "rss"

    if not headlines:
        headlines = MOCK_NEWS_BY_TICKER.get(
            ticker,
            [
                "Company updates awaited as markets assess sector outlook",
                "Analysts review latest quarterly trends for the stock",
                "Investors watch macro signals before taking fresh positions",
                "Trading volumes rise ahead of upcoming management commentary",
                "Mixed cues keep sentiment balanced in the near term",
            ],
        )[:max_items]

    result_df = _score_headlines(headlines, scorer=scorer)
    result_df["news_source"] = source
    return result_df
