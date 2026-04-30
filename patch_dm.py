import re

with open('src/data/data_manager.py', 'r') as f:
    text = f.read()

# Add math import
if 'import math' not in text:
    text = text.replace('MOCK_NEWS_BY_TICKER', 'import math\n\nMOCK_NEWS_BY_TICKER')

# Add VWAP, BB, ATR
advanced_inds = """    price_df["SMA_20"] = price_df["Close"].rolling(20).mean()
    price_df["SMA_50"] = price_df["Close"].rolling(50).mean()
    price_df["EMA_20"] = price_df["Close"].ewm(span=20, adjust=False).mean()
    price_df["DAILY_RETURN_PCT"] = price_df["Close"].pct_change() * 100

    # Advanced Indicators
    price_df["STD_20"] = price_df["Close"].rolling(20).std()
    price_df["BB_UPPER"] = price_df["SMA_20"] + (2 * price_df["STD_20"])
    price_df["BB_LOWER"] = price_df["SMA_20"] - (2 * price_df["STD_20"])

    if "Volume" in price_df.columns:
        price_df["VWAP"] = (price_df["Close"] * price_df["Volume"]).cumsum() / (price_df["Volume"].cumsum() + 1e-9)

    if ta is not None:
        price_df["RSI_14"] = ta.rsi(price_df["Close"], length=14)
        macd_df = ta.macd(price_df["Close"], fast=12, slow=26, signal=9)
        
        # ATR computation via pandas_ta
        true_range = ta.true_range(price_df["High"], price_df["Low"], price_df["Close"])
        if true_range is not None:
            price_df["ATR_14"] = ta.sma(true_range, length=14)
        else:
            price_df["ATR_14"] = pd.NA

        if macd_df is not None and not macd_df.empty:"""

old_inds = """    price_df["SMA_20"] = price_df["Close"].rolling(20).mean()
    price_df["SMA_50"] = price_df["Close"].rolling(50).mean()
    price_df["EMA_20"] = price_df["Close"].ewm(span=20, adjust=False).mean()
    price_df["DAILY_RETURN_PCT"] = price_df["Close"].pct_change() * 100

    if ta is not None:
        price_df["RSI_14"] = ta.rsi(price_df["Close"], length=14)
        macd_df = ta.macd(price_df["Close"], fast=12, slow=26, signal=9)

        if macd_df is not None and not macd_df.empty:"""

text = text.replace(old_inds, advanced_inds)

new_chain = """def fetch_nifty_option_chain(symbol: str = "NIFTY") -> dict[str, Any]:
    \"\"\"Fetch NSE option-chain snapshot for NIFTY index derivatives.\"\"\"
    def generate_synthetic_chain() -> dict[str, Any]:
        try:
            spot = float(yf.Ticker("^NSEI", session=get_yfinance_session()).fast_info.get("lastPrice", 22000.0))
        except Exception:
            spot = 22000.0
            
        atm_strike = round(spot / 50) * 50
        strikes = [atm_strike + (i * 50) for i in range(-15, 16)]
        
        parsed_rows = []
        for strike in strikes:
            dist = strike - spot
            dist_pts = abs(dist)
            iv = 12.0 + (dist_pts / 200.0)
            ce_oi = max(1000, 2000000 - abs(dist - 300) * 2000) if dist > -200 else max(500, 500000 - dist_pts * 1000)
            pe_oi = max(1000, 2000000 - abs(dist + 300) * 2000) if dist < 200 else max(500, 500000 - dist_pts * 1000)
            gamma = 0.005 * math.exp(-0.5 * (dist_pts / 100.0)**2)
            tv = 150 * math.exp(-0.5 * (dist_pts / 200.0)**2)
            ce_price = max(0, spot - strike) + tv
            pe_price = max(0, strike - spot) + tv

            parsed_rows.append({
                "strike": float(strike),
                "expiry": (pd.Timestamp.now() + pd.Timedelta(days=4)).strftime("%d-%b-%Y"),
                "ce_oi": float(ce_oi), "pe_oi": float(pe_oi),
                "ce_change_oi": float(ce_oi * 0.1), "pe_change_oi": float(pe_oi * 0.1),
                "ce_iv": float(iv), "pe_iv": float(iv + (0.5 if dist < 0 else 0)),
                "ce_last": float(ce_price), "pe_last": float(pe_price),
                "ce_gamma": float(gamma), "pe_gamma": float(gamma)
            })

        chain_df = pd.DataFrame(parsed_rows).sort_values("strike").reset_index(drop=True)
        return {
            "symbol": symbol.upper(),
            "underlying_value": spot,
            "nearest_expiry": parsed_rows[0]["expiry"],
            "fetched_at": pd.Timestamp.utcnow().isoformat(),
            "chain_df": chain_df,
            "is_mock": True
        }

    session = curl_requests.Session(impersonate="chrome")
    headers = dict(NSE_HEADERS)
    headers["Accept"] = "application/json,text/plain,*/*"
    session.headers.update(headers)

    verify: str | bool = _REQUESTS_VERIFY
    api_url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol.upper()}"

    try:
        session.get("https://www.nseindia.com", timeout=5, verify=verify)
        session.get("https://www.nseindia.com/option-chain", timeout=5, verify=verify)
    except Exception:
        verify = False
        try:
            session.get("https://www.nseindia.com", timeout=5, verify=False)
            session.get("https://www.nseindia.com/option-chain", timeout=5, verify=False)
        except Exception:
            pass

    try:
        response = session.get(api_url, timeout=6, verify=verify)
        if response.status_code != 200 or not response.text.strip() or response.text.strip() == '{}':
            return generate_synthetic_chain()
        payload = response.json()
        if not payload or 'records' not in payload:
            return generate_synthetic_chain()
    except Exception:
        try:
            response = session.get(api_url, timeout=6, verify=False)
            if response.status_code != 200 or not response.text.strip() or response.text.strip() == '{}':
                return generate_synthetic_chain()
            payload = response.json()
            if not payload or 'records' not in payload:
                return generate_synthetic_chain()
        except Exception:
            return generate_synthetic_chain()

    # Process payload if we successfully got it
    records = payload.get("records", {}) if isinstance(payload, dict) else {}
    option_rows = records.get("data", []) if isinstance(records, dict) else []
    expiries = records.get("expiryDates", []) if isinstance(records, dict) else []
    nearest_expiry = str(expiries[0]) if expiries else None
    underlying_value = _safe_json_num(records.get("underlyingValue")) if isinstance(records, dict) else None

    parsed_rows = []
    for row in option_rows:
        if not isinstance(row, dict):
            continue
        expiry = str(row.get("expiryDate", ""))
        if nearest_expiry and expiry != nearest_expiry:
            continue

        strike = _safe_json_num(row.get("strikePrice"))
        if strike is None:
            continue

        ce_leg = row.get("CE") if isinstance(row.get("CE"), dict) else {}
        pe_leg = row.get("PE") if isinstance(row.get("PE"), dict) else {}

        parsed_rows.append({
            "strike": strike,
            "expiry": expiry,
            "ce_oi": _option_metric(ce_leg, "openInterest") or 0.0,
            "pe_oi": _option_metric(pe_leg, "openInterest") or 0.0,
            "ce_change_oi": _option_metric(ce_leg, "changeinOpenInterest") or 0.0,
            "pe_change_oi": _option_metric(pe_leg, "changeinOpenInterest") or 0.0,
            "ce_iv": _option_metric(ce_leg, "impliedVolatility"),
            "pe_iv": _option_metric(pe_leg, "impliedVolatility"),
            "ce_last": _option_metric(ce_leg, "lastPrice"),
            "pe_last": _option_metric(pe_leg, "lastPrice"),
            "ce_gamma": _option_metric(ce_leg, "gamma"),
            "pe_gamma": _option_metric(pe_leg, "gamma"),
        })

    if not parsed_rows:
        return generate_synthetic_chain()

    chain_df = pd.DataFrame(parsed_rows).sort_values("strike").reset_index(drop=True)
    return {
        "symbol": symbol.upper(),
        "underlying_value": underlying_value,
        "nearest_expiry": nearest_expiry,
        "fetched_at": pd.Timestamp.utcnow().isoformat(),
        "chain_df": chain_df,
    }"""

# regex replace the entire fetch_nifty_option_chain
pattern = r'def fetch_nifty_option_chain\(symbol: str = "NIFTY"\) -> dict\[str, Any\]:.*?(?=\n\n\ndef |\Z)'
text = re.sub(pattern, new_chain, text, flags=re.DOTALL)

with open('src/data/data_manager.py', 'w') as f:
    f.write(text)

print("Done patching.")
