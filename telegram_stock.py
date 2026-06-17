import os
import telebot
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from html import unescape

# ============================
# LOAD TELEGRAM TOKEN
# ============================
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
bot = telebot.TeleBot(BOT_TOKEN)

NEWS_DEBUG = os.getenv("NEWS_DEBUG", "0") == "1"


# ============================
# SCANNER FUNCTIONS
# ============================

def fix_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")]
    return df


def get_data(ticker):
    df_5m = yf.download(ticker, interval="5m", period="2d", progress=False)
    df_1h = yf.download(ticker, interval="1h", period="2wk", progress=False)
    return fix_df(df_5m), fix_df(df_1h)


def add_indicators(df):
    if df.empty:
        return df

    close = df["Close"]

    df["EMA9"] = close.ewm(span=9).mean()
    df["EMA21"] = close.ewm(span=21).mean()

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

    df["EMA12"] = close.ewm(span=12).mean()
    df["EMA26"] = close.ewm(span=26).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()

    low14 = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    df["StochK"] = (close - low14) / (high14 - low14).replace(0, np.nan) * 100
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


def volume_level(df):
    df = df.dropna()
    if df.empty:
        return "No Data"

    rv = df["Volume"].iloc[-1] / df["Volume"].rolling(20).mean().iloc[-1]

    if rv < 0.7:
        return "🔵 Low Volume"
    elif rv < 1.3:
        return "🟡 Normal Volume"
    else:
        return "🔥 High Volume"


def fear_greed_index():
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        r = requests.get(url, timeout=5).json()
        value = int(r["data"][0]["value"])
        classification = r["data"][0]["value_classification"]

        emoji = "😱" if value < 25 else "😟" if value < 45 else "😐" if value < 55 else "😊" if value < 75 else "🤩"
        return f"{emoji} {value} — {classification}"
    except:
        return "Fear & Greed: Unavailable"


def market_condition():
    nasdaq = fix_df(yf.download("^IXIC", period="2d", interval="1d", progress=False))
    nyse = fix_df(yf.download("^NYA", period="2d", interval="1d", progress=False))

    def check(df):
        if df.empty:
            return "No Data"
        return "🟢 Green Day" if df["Close"].iloc[-1] > df["Open"].iloc[-1] else "🔴 Red Day"

    return {"NASDAQ": check(nasdaq), "NYSE": check(nyse)}


def fetch_rss_titles(url, max_items=5):
    try:
        r = requests.get(url, timeout=5)
        root = ET.fromstring(r.content)
        titles = []
        for item in root.findall(".//item")[:max_items]:
            t = item.find("title")
            if t is not None:
                titles.append("• " + unescape(t.text.strip()))
        return titles
    except:
        return []


def stock_news(ticker):
    try:
        t = yf.Ticker(ticker)
        news = t.news
        if news:
            return ["• " + n["title"] for n in news[:5]]
    except:
        pass

    return fetch_rss_titles(f"https://finance.yahoo.com/rss/headline?s={ticker}")


def run_scanner(ticker):
    output = []

    # Daily stats
    df_raw = yf.download(ticker, interval="1d", period="1mo", progress=False)
    df = fix_df(df_raw).dropna()

    if df.empty:
        return "No data available."

    output.append(f"=== Daily Price Stats — {ticker} ===")
    output.append(f"Current Price: {df['Close'].iloc[-1]:.2f}")
    output.append(f"Today Open: {df['Open'].iloc[-1]:.2f}")

    # 5m + 1h
    df5, df1 = get_data(ticker)
    df5 = add_indicators(df5)
    df1 = add_indicators(df1)

    output.append("\n=== Market Conditions ===")
    output.append("Volatility: " + volatility_regime(df5))
    output.append("Volume: " + volume_level(df5))
    output.append("Fear & Greed: " + fear_greed_index())

    mc = market_condition()
    output.append(f"NASDAQ: {mc['NASDAQ']}")
    output.append(f"NYSE: {mc['NYSE']}")

    output.append("\n=== News ===")
    for h in stock_news(ticker):
        output.append(h)

    return "\n".join(output)


# ============================
# TELEGRAM BOT
# ============================

@bot.message_handler(commands=['start'])
def start(msg):
    bot.reply_to(msg, "Send me any ticker (TSLA, NVDA, AAPL, SPY, BTC-USD).")


@bot.message_handler(func=lambda m: True)
def handle_message(msg):
    ticker = msg.text.strip().upper()
    bot.reply_to(msg, f"⏳ Scanning {ticker}...")

    try:
        result = run_scanner(ticker)

        # Telegram max message size = 4096 chars
        if len(result) > 4000:
            for i in range(0, len(result), 4000):
                bot.send_message(msg.chat.id, result[i:i+4000])
        else:
            bot.send_message(msg.chat.id, result)

    except Exception as e:
        bot.send_message(msg.chat.id, f"❌ Error: {e}")


bot.polling(none_stop=True)
