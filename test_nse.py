import requests
import json
import random

def fetch_nse():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    s.get("https://www.nseindia.com")
    r = s.get("https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY")
    return r.text

print(fetch_nse()[:100])
