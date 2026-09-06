from __future__ import annotations

import argparse
import csv
import json
import re
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from xml.etree import ElementTree as ET

import pandas as pd
import requests
import yfinance as yf

DEFAULT_UNIVERSE_FILE = Path(__file__).with_name("market_universe.txt")

POSITIVE_WORDS = {
    "gain",
    "rally",
    "upgrade",
    "strong",
    "record",
    "beat",
    "growth",
    "expansion",
    "approval",
    "buy",
    "target",
    "partnership",
    "momentum",
    "surge",
    "breakout",
    "upbeat",
    "profit",
    "outperform",
    "positive",
}
NEGATIVE_WORDS = {
    "loss",
    "drop",
    "downgrade",
    "weak",
    "delay",
    "risk",
    "fall",
    "shortfall",
    "warning",
    "sell",
    "pressure",
    "slowdown",
    "decline",
    "negative",
    "underperform",
    "downtick",
    "retrench",
}


def normalize_symbol(symbol: str) -> str:
    clean = symbol.strip().upper().replace(".NS", "").replace(".BSE", "")
    if not clean:
        raise ValueError("Empty stock symbol")
    return f"{clean}.NS"


def read_universe(path: Path | str | None = None) -> list[str]:
    source = Path(path) if path else DEFAULT_UNIVERSE_FILE
    if not source.exists():
        raise FileNotFoundError(f"Universe file not found: {source}")
    symbols: list[str] = []
    with source.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            for part in re.split(r"[\n,]+", line):
                clean = part.strip()
                if clean:
                    symbols.append(clean)
    return sorted(set(symbols))


def fetch_news_headlines(symbol: str, days: int = 30, max_items: int = 8) -> list[str]:
    query = f"{symbol.replace('.NS', '').replace('.BSE', '')} NSE stock"
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        response = requests.get(url, timeout=12)
        response.raise_for_status()
    except requests.RequestException:
        return []

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError:
        return []

    headlines: list[str] = []
    for item in root.findall("./channel/item")[:max_items]:
        title = item.findtext("title")
        if title:
            headlines.append(title)
    return headlines


def news_sentiment_score(symbol: str, days: int = 30) -> tuple[int, list[str]]:
    headlines = fetch_news_headlines(symbol, days=days)
    if not headlines:
        return 0, []
    score = 0
    for headline in headlines:
        text = headline.lower()
        pos_hits = sum(1 for word in POSITIVE_WORDS if word in text)
        neg_hits = sum(1 for word in NEGATIVE_WORDS if word in text)
        score += pos_hits - neg_hits
    return score, headlines


def compute_rsi(series: pd.Series, period: int = 14) -> float:
    if len(series) < period + 1:
        return 50.0
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window=period, min_periods=period).mean().iloc[-1]
    avg_loss = losses.rolling(window=period, min_periods=period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def analyze_daily_symbol(symbol: str, min_volume: int = 250_000, news_days: int = 30) -> dict[str, Any] | None:
    normalized = normalize_symbol(symbol)
    try:
        ticker = yf.Ticker(normalized)
        history = ticker.history(period="3mo", interval="1d", auto_adjust=True, actions=False)
    except Exception:
        return None

    if history.empty:
        return None

    close = pd.to_numeric(history["Close"], errors="coerce").dropna()
    volume = pd.to_numeric(history["Volume"], errors="coerce").dropna()
    if close.empty or volume.empty:
        return None

    last_price = float(close.iloc[-1])
    sma_20 = float(close.tail(20).mean()) if len(close) >= 20 else float(close.mean())
    sma_50 = float(close.tail(50).mean()) if len(close) >= 50 else float(close.mean())
    volume_ratio = float(volume.iloc[-1] / volume.tail(20).mean()) if len(volume) >= 20 else 1.0
    rsi = compute_rsi(close, 14)
    news_score, headlines = news_sentiment_score(normalized, days=news_days)

    score = 40
    if last_price > sma_20:
        score += 18
    else:
        score -= 12
    if sma_20 > sma_50:
        score += 12
    else:
        score -= 8
    if 45 <= rsi <= 70:
        score += 12
    elif rsi < 30:
        score -= 10
    if volume_ratio >= 1.2:
        score += 12
    if volume.iloc[-1] >= min_volume:
        score += 8
    score += min(max(news_score, -10), 15)

    signal = "HOLD"
    if score >= 75 and last_price > sma_20 and sma_20 > sma_50:
        signal = "BUY"
    elif score <= 35 or last_price < sma_20:
        signal = "SELL"

    return {
        "symbol": normalized.replace(".NS", ""),
        "status": "OK",
        "last_price": round(last_price, 2),
        "sma_20": round(sma_20, 2),
        "sma_50": round(sma_50, 2),
        "rsi_14": round(rsi, 2),
        "daily_change_pct": round(((last_price / close.iloc[-2]) - 1) * 100, 2) if len(close) >= 2 else 0.0,
        "volume": int(volume.iloc[-1]),
        "volume_ratio": round(volume_ratio, 2),
        "news_score": news_score,
        "signal": signal,
        "analysis_score": max(0, min(100, int(score))),
        "trend": "bullish" if last_price > sma_20 and sma_20 > sma_50 else "bearish" if last_price < sma_20 and sma_20 < sma_50 else "neutral",
        "news_headlines": headlines[:3],
    }


def rank_top_stocks(symbols: list[str], top_n: int = 10, min_volume: int = 250_000, news_days: int = 30) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for symbol in symbols:
        entry = analyze_daily_symbol(symbol, min_volume=min_volume, news_days=news_days)
        if entry is not None:
            results.append(entry)
    ranked = sorted(results, key=lambda item: item["analysis_score"], reverse=True)
    final_list = ranked[: max(1, top_n)]
    return {
        "market": "India / NSE + BSE liquid universe",
        "generated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "top_n": len(final_list),
        "results": final_list,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("symbol,analysis_score,signal,last_price,news_score\n", encoding="utf-8")
        return
    fieldnames = [
        "symbol",
        "analysis_score",
        "signal",
        "last_price",
        "sma_20",
        "sma_50",
        "rsi_14",
        "daily_change_pct",
        "volume",
        "volume_ratio",
        "news_score",
        "trend",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def send_telegram_message(bot_token: str, chat_id: str, message: str) -> bool:
    if not bot_token or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
        response = requests.post(url, data=payload, timeout=15)
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


def send_email_alert(
    smtp_server: str,
    smtp_port: int,
    sender_email: str,
    sender_password: str,
    recipient_emails: list[str],
    subject: str,
    body: str,
) -> bool:
    if not smtp_server or not sender_email or not recipient_emails or not sender_password:
        return False
    try:
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = sender_email
        message["To"] = ", ".join(recipient_emails)
        message.set_content(body)
        with smtplib.SMTP(smtp_server, smtp_port, timeout=20) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(message)
        return True
    except (OSError, smtplib.SMTPException):
        return False


def build_alert_message(report: dict[str, Any]) -> str:
    lines = [
        "<b>Indian pre-market top 10</b>",
        f"Generated: {report.get('generated_at', 'n/a')}",
    ]
    for idx, item in enumerate(report.get("results", []), start=1):
        symbol = item.get("symbol", "N/A")
        score = item.get("analysis_score", 0)
        signal = item.get("signal", "HOLD")
        price = item.get("last_price", "N/A")
        reason = item.get("trend", "neutral")
        lines.append(f"{idx}. <b>{symbol}</b> | {signal} | Score {score} | ₹{price} | {reason}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-market Indian stock screener with news sentiment scoring")
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE_FILE, help="TXT file with one Indian stock symbol per line")
    parser.add_argument("--top", type=int, default=10, help="Number of stocks to keep in the final recommendation list")
    parser.add_argument("--min-volume", type=int, default=250_000, help="Minimum average daily volume filter")
    parser.add_argument("--news-days", type=int, default=30, help="How many days of recent news to consider")
    parser.add_argument("--output", type=Path, default=Path("top_10_stocks.json"), help="Where to write the final JSON output")
    parser.add_argument("--csv-output", type=Path, default=Path("top_10_stocks.csv"), help="Optional CSV export path")
    parser.add_argument("--telegram-token", default="", help="Telegram bot token for sending alerts")
    parser.add_argument("--telegram-chat-id", default="", help="Telegram chat id for alerts")
    parser.add_argument("--smtp-server", default="", help="SMTP server hostname for email alerts")
    parser.add_argument("--smtp-port", type=int, default=587, help="SMTP server port")
    parser.add_argument("--smtp-user", default="", help="SMTP username")
    parser.add_argument("--smtp-password", default="", help="SMTP password")
    parser.add_argument("--email-from", default="", help="Sender email address for alert emails")
    parser.add_argument("--email-to", nargs="*", default=[], help="List of recipient email addresses for alerts")
    args = parser.parse_args()

    symbols = read_universe(args.universe)
    report = rank_top_stocks(symbols, top_n=args.top, min_volume=args.min_volume, news_days=args.news_days)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(args.csv_output, report["results"])

    alert_message = build_alert_message(report)
    if args.telegram_token and args.telegram_chat_id:
        send_telegram_message(args.telegram_token, args.telegram_chat_id, alert_message)
    if args.smtp_server and args.smtp_user and args.smtp_password and args.email_from and args.email_to:
        send_email_alert(
            smtp_server=args.smtp_server,
            smtp_port=args.smtp_port,
            sender_email=args.email_from,
            sender_password=args.smtp_password,
            recipient_emails=args.email_to,
            subject="Indian market top 10 alert",
            body=alert_message.replace("<b>", "").replace("</b>", ""),
        )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
