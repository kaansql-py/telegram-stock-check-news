import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from html import unescape

NEWS_DEBUG = os.getenv("NEWS_DEBUG", "0") == "1"


# -----------------------------
# FIX YFINANCE MULTI-INDEX
# -----------------------------
def fix_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")]
    return df


# -----------------------------
# GET 5m (2 days) + 1h (2 weeks)
# -----------------------------
def get_data(ticker):
    df_5m = yf.download(ticker, interval="5m", period="2d", progress=False)
    df_1h = yf.download(ticker, interval="1h", period="2wk", progress=False)
    return fix_df(df_5m), fix_df(df_1h)


# -----------------------------
# ADD INDICATORS
# -----------------------------
def add_indicators(df):
    if df.empty:
        return df

    close = df["Close"]

    df["EMA9"] = close.ewm(span=9, min_periods=9).mean()
    df["EMA21"] = close.ewm(span=21, min_periods=21).mean()

    df["CumVol"] = df["Volume"].cumsum()
    df["CumPV"] = (close * df["Volume"]).cumsum()
    df["VWAP"] = df["CumPV"] / df["CumVol"].replace(0, np.nan)

    df["Momentum"] = close - close.shift(10)

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["EMA12"] = close.ewm(span=12, min_periods=12).mean()
    df["EMA26"] = close.ewm(span=26, min_periods=26).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, min_periods=9).mean()

    low14 = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    range14 = (high14 - low14).replace(0, np.nan)
    df["StochK"] = (close - low14) / range14 * 100
    df["StochD"] = df["StochK"].rolling(3).mean()

    df["MA20"] = close.rolling(20).mean()
    df["STD20"] = close.rolling(20).std()
    df["BB_Upper"] = df["MA20"] + 2 * df["STD20"]
    df["BB_Lower"] = df["MA20"] - 2 * df["STD20"]

    high = df["High"]
    low = df["Low"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    df["BullFVG"] = ((df["Low"] > df["High"].shift(2))).astype(int)
    df["BearFVG"] = ((df["High"] < df["Low"].shift(2))).astype(int)

    return df


# -----------------------------
# VOLATILITY REGIME (ATR Percentile)
# -----------------------------
def volatility_regime(df):
    df = df.dropna()
    if df.empty:
        return "No Data"

    atr_series = df["ATR14"].dropna()
    if len(atr_series) < 10:
        return "Not Enough Data"

    last_atr = atr_series.iloc[-1]
    percentile = (atr_series < last_atr).mean()

    if percentile < 0.33:
        return "🔵 Low Volatility"
    elif percentile < 0.66:
        return "🟡 Medium Volatility"
    else:
        return "🔥 High Volatility"


# -----------------------------
# VOLUME LEVEL (Relative Volume)
# -----------------------------
def volume_level(df):
    df = df.dropna()
    if df.empty:
        return "No Data"

    if len(df["Volume"].rolling(20).mean().dropna()) == 0:
        return "No Data"

    rv = df["Volume"].iloc[-1] / df["Volume"].rolling(20).mean().iloc[-1]

    if rv < 0.7:
        return "🔵 Low Volume"
    elif rv < 1.3:
        return "🟡 Normal Volume"
    else:
        return "🔥 High Volume"


# -----------------------------
# FEAR & GREED INDEX (Alternative.me API)
# -----------------------------
def fear_greed_index():
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        r = requests.get(url, timeout=5).json()

        value = int(r["data"][0]["value"])
        classification = r["data"][0]["value_classification"]

        emoji = "😱" if value < 25 else "😟" if value < 45 else "😐" if value < 55 else "😊" if value < 75 else "🤩"

        return f"{emoji} {value} — {classification}"

    except Exception:
        return "Fear & Greed: Data Unavailable"


# -----------------------------
# MARKET CONDITION (NASDAQ + NYSE)
# -----------------------------
def market_condition():
    nasdaq_raw = yf.download("^IXIC", period="2d", interval="1d", progress=False)
    nyse_raw = yf.download("^NYA", period="2d", interval="1d", progress=False)

    nasdaq = fix_df(nasdaq_raw) if not nasdaq_raw.empty else nasdaq_raw
    nyse = fix_df(nyse_raw) if not nyse_raw.empty else nyse_raw

    def check(df):
        if df.empty:
            return "No Data"
        open_ = float(df["Open"].iloc[-1])
        close_ = float(df["Close"].iloc[-1])
        return "🟢 Green Day" if close_ > open_ else "🔴 Red Day"

    return {
        "NASDAQ": check(nasdaq),
        "NYSE": check(nyse)
    }


# -----------------------------
# HELPER: parse RSS safely and return titles
# -----------------------------
def fetch_rss_titles(url, max_items=5, timeout=6):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        titles = []
        for item in root.findall(".//item")[:max_items]:
            title_el = item.find("title")
            if title_el is not None and title_el.text:
                titles.append(unescape(title_el.text.strip()))
        if not titles:
            for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry")[:max_items]:
                title_el = entry.find("{http://www.w3.org/2005/Atom}title")
                if title_el is not None and title_el.text:
                    titles.append(unescape(title_el.text.strip()))
        return titles
    except Exception:
        return []


# -----------------------------
# STOCK NEWS (robust, layered)
# -----------------------------
def stock_news(ticker, max_headlines=5):
    headlines = []

    # 1) Try yfinance Ticker.news (fast, often reliable)
    try:
        t = yf.Ticker(ticker)
        news_list = t.news  # list of dicts with 'title' and 'link'
        if NEWS_DEBUG:
            print("DEBUG: yfinance news count:", len(news_list) if news_list is not None else 0)
        if news_list:
            for item in news_list[:max_headlines]:
                title = item.get("title") or item.get("headline") or None
                if title:
                    headlines.append("• " + title)
    except Exception as e:
        if NEWS_DEBUG:
            print("DEBUG: yfinance news error:", repr(e))

    # 2) Yahoo Finance RSS fallback
    if len(headlines) < max_headlines:
        try:
            yf_rss = f"https://finance.yahoo.com/rss/headline?s={ticker}"
            if NEWS_DEBUG:
                print("DEBUG: fetching Yahoo RSS:", yf_rss)
            titles = fetch_rss_titles(yf_rss, max_items=max_headlines)
            for t in titles:
                if len(headlines) >= max_headlines:
                    break
                headlines.append("• " + t)
        except Exception as e:
            if NEWS_DEBUG:
                print("DEBUG: yahoo rss error:", repr(e))

    # 3) Seeking Alpha combined feed
    if len(headlines) < max_headlines:
        try:
            sa_rss = f"https://seekingalpha.com/api/sa/combined/{ticker}.xml"
            if NEWS_DEBUG:
                print("DEBUG: fetching Seeking Alpha RSS:", sa_rss)
            titles = fetch_rss_titles(sa_rss, max_items=max_headlines)
            for t in titles:
                if len(headlines) >= max_headlines:
                    break
                headlines.append("• " + t)
        except Exception as e:
            if NEWS_DEBUG:
                print("DEBUG: seekingalpha rss error:", repr(e))

    # 4) Generic financial feeds (CNBC, Reuters)
    if len(headlines) < max_headlines:
        for feed in [
            "https://www.cnbc.com/id/100003114/device/rss/rss.html",
            "https://www.reuters.com/finance/markets/us"
        ]:
            if len(headlines) >= max_headlines:
                break
            titles = fetch_rss_titles(feed, max_items=max_headlines)
            for t in titles:
                if len(headlines) >= max_headlines:
                    break
                headlines.append("• " + t)

    # 5) Optional: NewsAPI (requires NEWSAPI_KEY env var)
    if len(headlines) < max_headlines:
        api_key = os.getenv("NEWSAPI_KEY")
        if api_key:
            try:
                params = {
                    "q": f"{ticker} OR Tesla OR TSLA",
                    "language": "en",
                    "pageSize": max_headlines - len(headlines),
                    "apiKey": api_key,
                }
                r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=6)
                r.raise_for_status()
                data = r.json()
                for art in data.get("articles", [])[: max_headlines - len(headlines)]:
                    title = art.get("title")
                    if title:
                        headlines.append("• " + title)
                if NEWS_DEBUG:
                    print("DEBUG: NewsAPI returned", len(data.get("articles", [])))
            except Exception as e:
                if NEWS_DEBUG:
                    print("DEBUG: NewsAPI error:", repr(e))

    if not headlines:
        return ["News fetch failed"]

    # dedupe and trim
    seen = set()
    out = []
    for h in headlines:
        if h not in seen:
            seen.add(h)
            out.append(h)
        if len(out) >= max_headlines:
            break

    return out


# -----------------------------
# DAILY PRICE STATS
# -----------------------------
def daily_stats(ticker):
    df_raw = yf.download(ticker, interval="1d", period="1mo", progress=False)
    df = fix_df(df_raw).dropna()

    if df.empty or len(df) < 2:
        print("\nNo daily data available.")
        return

    current_price = float(df["Close"].iloc[-1])
    today_open = float(df["Open"].iloc[-1])
    prev_open = float(df["Open"].iloc[-2])
    prev_close = float(df["Close"].iloc[-2])

    df["NetChange"] = df["Close"] - df["Open"]
    df["AbsChange"] = (df["Close"] - df["Open"]).abs()
    df["Range"] = df["High"] - df["Low"]

    avg_net = float(df["NetChange"].mean())
    avg_abs = float(df["AbsChange"].mean())
    avg_range = float(df["Range"].mean())

    print(f"\n=== Daily Price Stats (1 Month) — {ticker} ===")
    print("Metric                      | Value")
    print("----------------------------|----------------")
    print(f"Current Price               | {current_price:.2f}")
    print(f"Today Open                  | {today_open:.2f}")
    print(f"Prev Day Open               | {prev_open:.2f}")
    print(f"Prev Day Close              | {prev_close:.2f}")
    print(f"Avg Net Change (1mo)        | {avg_net:.2f}")
    print(f"Avg Abs Daily Move (1mo)    | {avg_abs:.2f}")
    print(f"Avg High-Low Range (1mo)    | {avg_range:.2f}")


# -----------------------------
# COMBINED 5m + 1h TABLE
# -----------------------------
def combined_table(df5, df1, ticker):
    df5 = df5.dropna()
    df1 = df1.dropna()

    if df5.empty or df1.empty:
        print("\nNo data available.")
        return

    last5 = df5.iloc[-1]
    last1 = df1.iloc[-1]

    def e(val5_bull, val5_bear, val1_bull, val1_bear):
        sig5 = "🟢" if val5_bull else "🔴" if val5_bear else "⚪"
        sig1 = "🟢" if val1_bull else "🔴" if val1_bear else "⚪"
        return sig5, sig1

    rows = [
        ("Trend (EMA)",
         *e(last5["EMA9"] > last5["EMA21"], last5["EMA9"] < last5["EMA21"],
            last1["EMA9"] > last1["EMA21"], last1["EMA9"] < last1["EMA21"])),

        ("VWAP",
         *e(last5["Close"] > last5["VWAP"], last5["Close"] < last5["VWAP"],
            last1["Close"] > last1["VWAP"], last1["Close"] < last1["VWAP"])),

        ("Momentum",
         *e(last5["Momentum"] > 0, last5["Momentum"] < 0,
            last1["Momentum"] > 0, last1["Momentum"] < 0)),

        ("RSI",
         *e(last5["RSI"] < 30, last5["RSI"] > 70,
            last1["RSI"] < 30, last1["RSI"] > 70)),

        ("MACD",
         *e(last5["MACD"] > last5["MACD_Signal"], last5["MACD"] < last5["MACD_Signal"],
            last1["MACD"] > last1["MACD_Signal"], last1["MACD"] < last1["MACD_Signal"])),

        ("Stoch",
         *e(last5["StochK"] < 20, last5["StochK"] > 80,
            last1["StochK"] < 20, last1["StochK"] > 80)),

        ("Bollinger",
         *e(last5["Close"] > last5["BB_Upper"], last5["Close"] < last5["BB_Lower"],
            last1["Close"] > last1["BB_Upper"], last1["Close"] < last1["BB_Lower"])),

        ("Bull FVG",
         "🟢" if last5["BullFVG"] else "⚪",
         "🟢" if last1["BullFVG"] else "⚪"),

        ("Bear FVG",
         "🔴" if last5["BearFVG"] else "⚪",
         "🔴" if last1["BearFVG"] else "⚪"),
    ]

    print(f"\n=== Indicator Summary — {ticker} ===")
    print("Indicator       | 5m Signal | 1h Signal")
    print("----------------|-----------|-----------")
    for name, sig5, sig1 in rows:
        print(f"{name:<15} | {sig5:<9} | {sig1}")


# -----------------------------
# MAIN
# -----------------------------
def main(ticker):
    daily_stats(ticker)

    df_5m, df_1h = get_data(ticker)

    df_5m = add_indicators(df_5m)
    df_1h = add_indicators(df_1h)

    print(f"\n=== Market Conditions — {ticker} ===")
    print("Volatility Regime:", volatility_regime(df_5m))
    print("Volume Level:", volume_level(df_5m))
    print("Fear & Greed Index:", fear_greed_index())

    mc = market_condition()
    print("\n=== Stock Market Condition ===")
    print("NASDAQ:", mc["NASDAQ"])
    print("NYSE:", mc["NYSE"])

    print(f"\n=== Recent News for {ticker} ===")
    for h in stock_news(ticker):
        print(h)

    combined_table(df_5m, df_1h, ticker)


if __name__ == "__main__":
    import sys
    ticker = "TSLA"
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    main(ticker)
