
import os
import time
import threading
import requests
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="NIFTY + BANKNIFTY AI Bot",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NIFTY + BANKNIFTY Trading Assistant")
st.caption("Educational prototype | Paper Trading Only")

REQUIRED = {
    "timestamp", "symbol", "timeframe",
    "open", "high", "low", "close", "volume"
}


# =========================
# TELEGRAM CONFIG
# =========================

def get_telegram_config():
    try:
        token = st.secrets["TELEGRAM_BOT_TOKEN"]
        chat_id = st.secrets["TELEGRAM_CHAT_ID"]
    except Exception:
        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    return str(token).strip(), str(chat_id).strip()


def send_telegram(message):
    token, chat_id = get_telegram_config()

    if not token or not chat_id:
        return False, "Telegram credentials missing."

    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message},
            timeout=15
        )
        data = response.json()

        if not data.get("ok"):
            return False, str(data.get("description", "Telegram error"))

        return True, "Telegram message sent!"

    except Exception as exc:
        return False, str(exc)


# =========================
# TELEGRAM LISTENER
# =========================

def check_telegram(token):
    try:
        response = requests.get(
            f"https://api.telegram.org/bot{token}/getMe",
            timeout=10
        )
        data = response.json()

        if not data.get("ok"):
            return False, "Invalid Telegram token."

        webhook = requests.get(
            f"https://api.telegram.org/bot{token}/getWebhookInfo",
            timeout=10
        ).json()

        if webhook.get("result", {}).get("url"):
            return False, "Webhook active. Polling cannot start."

        return True, "Connected to @" + data["result"]["username"]

    except Exception as exc:
        return False, str(exc)


@st.cache_resource
def start_telegram_listener(token, allowed_chat):

    def listen():
        offset = None
        base = f"https://api.telegram.org/bot{token}"

        print("Telegram listener started")

        while True:
            try:
                params = {"timeout": 25}

                if offset is not None:
                    params["offset"] = offset

                response = requests.get(
                    base + "/getUpdates",
                    params=params,
                    timeout=35
                )
                response.raise_for_status()
                data = response.json()

                if not data.get("ok"):
                    print("Telegram API error:", data)
                    time.sleep(5)
                    continue

                for update in data.get("result", []):
                    offset = update["update_id"] + 1

                    message = update.get("message", {})
                    chat_id = str(
                        message.get("chat", {}).get("id", "")
                    )

                    text = message.get("text", "").strip()

                    if not text or chat_id != allowed_chat:
                        continue

                    command = text.split()[0].lower().split("@")[0]

                    replies = {
                        "/start": "Welcome! Your NIFTY BANKNIFTY bot is connected. Send /help.",
                        "/help": "Commands:\n/start\n/help\n/status\n/nifty\n/banknifty",
                        "/status": "Bot online. Paper trading only. Live market data is not connected.",
                        "/nifty": "NIFTY: Upload CSV on the dashboard for analysis. No live signal available.",
                        "/banknifty": "BANKNIFTY: Upload CSV on the dashboard for analysis. No live signal available."
                    }

                    reply = replies.get(
                        command,
                        "Unknown command. Send /help."
                    )

                    sent = requests.post(
                        base + "/sendMessage",
                        json={
                            "chat_id": chat_id,
                            "text": reply
                        },
                        timeout=15
                    )

                    sent.raise_for_status()
                    print("Telegram reply sent:", command)

            except Exception as exc:
                print("TELEGRAM ERROR:", repr(exc))
                time.sleep(5)

    thread = threading.Thread(
        target=listen,
        daemon=True
    )
    thread.start()
    return True


token, allowed_chat = get_telegram_config()

if token and allowed_chat:
    ok, status = check_telegram(token)

    if ok:
        st.success(status)
        start_telegram_listener(token, allowed_chat)
    else:
        st.error(status)
else:
    st.warning("Telegram Secrets are missing.")


# =========================
# INDICATORS
# =========================

def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0).ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = gain / loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def add_indicators(df):
    df = df.sort_values("timestamp").copy()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ema9"] = df["close"].ewm(
        span=9, adjust=False
    ).mean()

    df["ema21"] = df["close"].ewm(
        span=21, adjust=False
    ).mean()

    df["rsi14"] = calculate_rsi(df["close"])

    fast = df["close"].ewm(
        span=12, adjust=False
    ).mean()

    slow = df["close"].ewm(
        span=26, adjust=False
    ).mean()

    df["macd"] = fast - slow

    df["macd_signal"] = df["macd"].ewm(
        span=9, adjust=False
    ).mean()

    previous = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous).abs(),
            (df["low"] - previous).abs()
        ],
        axis=1
    ).max(axis=1)

    df["atr14"] = tr.rolling(14).mean()
    df["vol_avg20"] = df["volume"].rolling(20).mean()

    return df


# =========================
# SIGNAL GENERATION
# =========================

def make_signal(row, timeframe):
    score = 0
    reasons = []

    if row["ema9"] > row["ema21"]:
        score += 1
        reasons.append("EMA bullish")

    elif row["ema9"] < row["ema21"]:
        score -= 1
        reasons.append("EMA bearish")

    if row["macd"] > row["macd_signal"]:
        score += 1
        reasons.append("MACD bullish")

    elif row["macd"] < row["macd_signal"]:
        score -= 1
        reasons.append("MACD bearish")

    if 52 <= row["rsi14"] <= 68:
        score += 1
        reasons.append("RSI bullish")

    elif 32 <= row["rsi14"] <= 48:
        score -= 1
        reasons.append("RSI bearish")

    if score >= 2:
        signal = "BUY"
    elif score <= -2:
        signal = "SELL"
    else:
        signal = "NO TRADE"

    price = float(row["close"])
    atr = row["atr14"]

    if pd.isna(atr) or atr <= 0:
        sl = t1 = t2 = None

    elif signal == "BUY":
        sl = price - 1.2 * atr
        t1 = price + 1.8 * atr
        t2 = price + 2.4 * atr

    elif signal == "SELL":
        sl = price + 1.2 * atr
        t1 = price - 1.8 * atr
        t2 = price - 2.4 * atr

    else:
        sl = t1 = t2 = None

    return {
        "signal": signal,
        "score": score,
        "entry": price,
        "stop_loss": sl,
        "target_1": t1,
        "target_2": t2,
        "reason": ", ".join(reasons),
        "timeframe": timeframe
    }


# =========================
# DASHBOARD
# =========================

st.info(
    "CSV-based analysis only. Live market feed and broker API are not connected."
)

uploaded = st.file_uploader(
    "Upload NIFTY / BANKNIFTY OHLCV CSV",
    type=["csv"]
)

st.caption(
    "Required columns: timestamp, symbol, timeframe, open, high, low, close, volume"
)


# =========================
# CSV ANALYSIS
# =========================

if uploaded:

    try:
        raw = pd.read_csv(uploaded)

        missing = REQUIRED - set(raw.columns)

        if missing:
            st.error("Missing columns: " + ", ".join(missing))
            st.stop()

        raw["timestamp"] = pd.to_datetime(
            raw["timestamp"], errors="coerce"
        )

        raw["symbol"] = (
            raw["symbol"].astype(str).str.upper().str.strip()
        )

        raw["timeframe"] = (
            raw["timeframe"].astype(str).str.strip()
        )

        for col in ["open", "high", "low", "close", "volume"]:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

        raw = raw.dropna(
            subset=["timestamp", "open", "high", "low", "close"]
        )

        symbols = [
            s for s in ["NIFTY", "BANKNIFTY"]
            if s in raw["symbol"].unique()
        ]

        if not symbols:
            st.error("NIFTY or BANKNIFTY data not found.")
            st.stop()

        selected_symbol = st.selectbox(
            "Select Index", symbols
        )

        timeframes = sorted(
            raw.loc[
                raw["symbol"] == selected_symbol,
                "timeframe"
            ].dropna().unique()
        )

        if not timeframes:
            st.error("No timeframe found.")
            st.stop()

        selected_tf = st.selectbox(
            "Select Timeframe", timeframes
        )

        data = raw[
            (raw["symbol"] == selected_symbol)
            & (raw["timeframe"] == selected_tf)
        ].copy()

        if len(data) < 35:
            st.warning("Please upload at least 35 candles.")

        if len(data) < 2:
            st.stop()

        ind = add_indicators(data)
        last = ind.iloc[-1]

        if last[
            ["ema9", "ema21", "rsi14", "macd", "macd_signal"]
        ].isna().any():
            st.warning("Indicators not ready. Upload more candles.")
            st.stop()

        result = make_signal(last, selected_tf)

        a, b, c = st.columns(3)

        a.metric("Latest Price", f'{result["entry"]:,.2f}')
        b.metric("Signal", result["signal"])
        c.metric("Score", result["score"])

        st.subheader("Paper Trading Plan")

        plan = pd.DataFrame([{
            "Index": selected_symbol,
            "Timeframe": selected_tf,
            "Signal": result["signal"],
            "Entry": result["entry"],
            "Stop Loss": result["stop_loss"],
            "Target 1": result["target_1"],
            "Target 2": result["target_2"]
        }])

        st.dataframe(plan, use_container_width=True)

        st.write("Signal reasons:", result["reason"])

        st.subheader("Price and EMA Chart")

        chart = ind.set_index("timestamp")[
            ["close", "ema9", "ema21"]
        ].tail(250)

        st.line_chart(chart)

        if st.button("Send Analysis to Telegram"):
            message = (
                f"📊 {selected_symbol} | {selected_tf}\n"
                f"Signal: {result['signal']}\n"
                f"Entry: {result['entry']:.2f}\n"
                f"Stop Loss: {result['stop_loss']}\n"
                f"Target 1: {result['target_1']}\n"
                f"Target 2: {result['target_2']}\n"
                f"Reason: {result['reason']}\n"
                "Paper Trading Only."
            )

            ok, note = send_telegram(message)

            if ok:
                st.success(note)
            else:
                st.error(note)

        with st.expander("Latest Indicator Values"):
            st.dataframe(ind.tail(20), use_container_width=True)

    except Exception as exc:
        st.error(f"CSV processing error: {exc}")

else:
    st.subheader("Trading Bot Information")

    st.markdown("""
    - EMA 9 and EMA 21
    - RSI 14
    - MACD
    - ATR 14
    - BUY / SELL / NO TRADE rules
    - Stop Loss and Targets
    - Telegram alerts
    """)

    st.warning(
        "No live trading or live market signals. "
        "Use paper trading and test historical data first."
    )
