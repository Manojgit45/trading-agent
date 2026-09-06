from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

DEFAULT_FILES = [
    Path("india_market_scan.json"),
    Path("live_analysis.json"),
    Path("paper_trading_report.json"),
]


def normalize_market_symbol(symbol: str) -> str:
    clean = symbol.strip().upper().replace(".NS", "").replace(".BSE", "")
    return f"{clean}.NS" if clean else symbol.strip().upper()


@st.cache_data
def load_analysis_json(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "results" in payload:
        rows = payload["results"]
    elif isinstance(payload, list):
        rows = payload
    else:
        return pd.DataFrame()

    cleaned = []
    for row in rows:
        if isinstance(row, dict):
            cleaned.append(row)

    df = pd.DataFrame(cleaned)
    if df.empty:
        return df

    expected = [
        "symbol",
        "status",
        "last_price",
        "sma_20",
        "sma_50",
        "rsi_14",
        "daily_change_pct",
        "volume",
        "signal",
        "analysis_score",
        "trend",
    ]
    for column in expected:
        if column not in df.columns:
            df[column] = None

    numeric_columns = ["last_price", "sma_20", "sma_50", "rsi_14", "daily_change_pct", "analysis_score", "volume"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["status"] = df["status"].fillna("UNKNOWN")
    df["signal"] = df["signal"].fillna("HOLD")
    df["trend"] = df["trend"].fillna("neutral")
    return df.sort_values(["analysis_score", "last_price"], ascending=[False, False]).reset_index(drop=True)


@st.cache_data
def load_symbol_history(symbol: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    ticker_name = normalize_market_symbol(symbol)
    ticker = yf.Ticker(ticker_name)
    hist = ticker.history(period=period, interval=interval, auto_adjust=True, actions=False)
    if hist.empty:
        return pd.DataFrame()
    frame = hist[["Close"]].copy().reset_index()
    frame.columns = ["Date", "Close"]
    frame["SMA_20"] = frame["Close"].rolling(window=20, min_periods=1).mean()
    frame["SMA_50"] = frame["Close"].rolling(window=50, min_periods=1).mean()
    return frame


st.set_page_config(page_title="Indian Stock Dashboard", page_icon="📈", layout="wide")
st.title("Indian Stock Analysis Dashboard")

with st.sidebar:
    st.header("Settings")
    data_file = st.selectbox(
        "Choose analysis file",
        options=[str(path) for path in DEFAULT_FILES if path.exists()],
        index=0 if DEFAULT_FILES[0].exists() else 0,
    )
    if not data_file:
        st.warning("No JSON report found. Generate a file with `agent.py --live --all` first.")
        st.stop()

    min_score = st.slider("Minimum analysis score", 0, 100, 0)
    signal_filter = st.selectbox("Signal", ["All", "BUY", "SELL", "HOLD"])
    search_term = st.text_input("Search symbol")

try:
    df = load_analysis_json(data_file)
except Exception as exc:  # pragma: no cover - user-facing validation
    st.error(f"Could not load {data_file}: {exc}")
    st.stop()

if df.empty:
    st.info("The selected JSON file is not a live analysis export. Please choose a file with a `results` array such as `live_analysis.json` or `india_market_scan.json`.")
    st.stop()

filtered = df.copy()
if signal_filter != "All":
    filtered = filtered[filtered["signal"] == signal_filter]
if min_score > 0:
    filtered = filtered[filtered["analysis_score"] >= min_score]
if search_term:
    filtered = filtered[filtered["symbol"].str.contains(search_term.strip().upper(), case=False, na=False)]

if filtered.empty:
    st.info("No stocks match the current filters.")
    st.stop()

signal_counts = filtered["signal"].value_counts().reindex(["BUY", "SELL", "HOLD"], fill_value=0)
trend_counts = filtered["trend"].value_counts()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Stocks", int(filtered["symbol"].nunique()))
col2.metric("Buy Signals", int(signal_counts.get("BUY", 0)))
col3.metric("Sell Signals", int(signal_counts.get("SELL", 0)))
col4.metric("Average Score", round(float(filtered["analysis_score"].mean()), 2) if not filtered["analysis_score"].empty else 0)

left, right = st.columns(2)
left.plotly_chart(
    px.bar(
        signal_counts.rename_axis("signal").reset_index(name="count"),
        x="signal",
        y="count",
        color="signal",
        color_discrete_map={"BUY": "#2ecc71", "SELL": "#e74c3c", "HOLD": "#f39c12"},
        title="Signal Count",
    ),
    width="stretch",
)
right.plotly_chart(
    px.bar(
        filtered.nlargest(15, "analysis_score")[["symbol", "analysis_score"]].sort_values("analysis_score", ascending=True),
        x="analysis_score",
        y="symbol",
        orientation="h",
        color="analysis_score",
        color_continuous_scale="Viridis",
        title="Top 15 by Score",
    ),
    width="stretch",
)

st.subheader("Market movers")
move_col1, move_col2 = st.columns(2)
move_col1.plotly_chart(
    px.bar(
        filtered.nlargest(10, "daily_change_pct")[["symbol", "daily_change_pct"]].sort_values("daily_change_pct", ascending=False),
        x="daily_change_pct",
        y="symbol",
        orientation="h",
        color="daily_change_pct",
        color_continuous_scale="RdYlGn",
        title="Top Gainers",
    ),
    width="stretch",
)
move_col2.plotly_chart(
    px.bar(
        filtered.nsmallest(10, "daily_change_pct")[["symbol", "daily_change_pct"]].sort_values("daily_change_pct", ascending=True),
        x="daily_change_pct",
        y="symbol",
        orientation="h",
        color="daily_change_pct",
        color_continuous_scale="RdYlGn_r",
        title="Top Losers",
    ),
    width="stretch",
)

st.subheader("Stock detail")
selected_symbol = st.selectbox("Inspect stock", filtered["symbol"].tolist())
selected_row = filtered[filtered["symbol"] == selected_symbol].iloc[0]
selected_cols = st.columns(5)
selected_cols[0].metric("Price", f"₹{selected_row['last_price']:.2f}" if pd.notna(selected_row['last_price']) else "N/A")
selected_cols[1].metric("Signal", selected_row["signal"])
selected_cols[2].metric("Score", int(selected_row["analysis_score"]))
selected_cols[3].metric("RSI(14)", round(float(selected_row["rsi_14"]), 2) if pd.notna(selected_row['rsi_14']) else 0)
selected_cols[4].metric("Daily %", f"{selected_row['daily_change_pct']:.2f}%" if pd.notna(selected_row['daily_change_pct']) else "N/A")

history = load_symbol_history(selected_symbol)
if not history.empty:
    st.plotly_chart(
        px.line(
            history,
            x="Date",
            y=["Close", "SMA_20", "SMA_50"],
            title=f"Recent price trend for {selected_symbol}",
            markers=True,
        ),
        width="stretch",
    )
else:
    st.info(f"No recent price history found for {selected_symbol}.")

csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download filtered CSV",
    data=csv_bytes,
    file_name="filtered_market_scan.csv",
    mime="text/csv",
)

st.subheader("Trend distribution")
if not trend_counts.empty:
    st.plotly_chart(
        px.pie(
            trend_counts.rename_axis("trend").reset_index(name="count"),
            names="trend",
            values="count",
            title="Trend Mix",
        ),
        width="stretch",
    )

st.subheader("Stock price history")
history_df = load_symbol_history(selected_symbol)
if not history_df.empty:
    st.line_chart(history_df.set_index("Date")[["Close", "SMA_20", "SMA_50"]])
else:
    st.info("No price history found for this stock.")

st.caption("Market data is from public Yahoo Finance feeds and should be treated as informational research, not execution advice.")
