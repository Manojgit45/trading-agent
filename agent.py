"""India-only NSE cash-equity paper-trading agent.

This program deliberately cannot submit live orders. It consumes daily OHLC CSV
files and records simulated trades so that a strategy can be validated safely.

It also supports a market-analysis mode for Indian stocks using public market
quotes from Yahoo Finance (near-real-time / delayed data, not broker-grade).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf


DEFAULT_INDIAN_SYMBOLS = [
    "RELIANCE",
    "TCS",
    "INFY",
    "HDFCBANK",
    "ICICIBANK",
    "SBIN",
    "ITC",
    "LTIM",
    "SUNPHARMA",
    "BHARTIARTL",
    "KOTAKBANK",
    "AXISBANK",
    "WIPRO",
    "TATAMOTORS",
    "HINDUNILVR",
    "HCLTECH",
    "ASIANPAINT",
    "ULTRACEMCO",
    "JSWSTEEL",
    "NTPC",
    "POWERGRID",
    "TATACONSUM",
    "TECHM",
    "INDUSINDBK",
    "MARUTI",
    "BAJAJFINANCE",
    "BAJAJAUTO",
    "HDFCLIFE",
    "ICICIPRULI",
    "M&M",
    "TITAN",
    "ONGC",
    "GRASIM",
    "CIPLA",
    "DRREDDY",
    "NESTLEIND",
    "DIVISLAB",
    "APOLLOHOSP",
    "SHRIRAMFIN",
    "LT",
    "SIEMENS",
    "EICHERMOT",
    "HEROMOTOCO",
    "TATASTEEL",
    "COALINDIA",
    "IOC",
    "BPCL",
    "GAIL",
    "HINDALCO",
    "UPL",
    "BEL",
    "BHEL",
    "BANKBARODA",
    "CANBK",
    "PNB",
    "ZOMATO",
    "TRENT",
    "DMART",
    "IRCTC",
    "INDIGO",
    "JUBLFOOD",
    "NAUKRI",
    "GODREJCP",
    "AUROPHARMA",
    "BIOCON",
    "BANDHANBNK",
    "IDFCFIRSTB",
    "YESBANK",
    "FEDERALBNK",
    "PIDILITIND",
    "MUTHOOTFIN",
    "LUPIN",
]

INDIAN_STOCK_WATCHLIST = DEFAULT_INDIAN_SYMBOLS


@dataclass(frozen=True)
class Candle:
    day: str
    close: float


@dataclass
class Position:
    symbol: str
    quantity: int
    entry_price: float
    entry_day: str


@dataclass
class Trade:
    day: str
    symbol: str
    side: str
    quantity: int
    price: float
    reason: str


@dataclass(frozen=True)
class RiskLimits:
    starting_cash: float = 100_000.0
    max_positions: int = 3
    max_position_fraction: float = 0.20
    stop_loss_fraction: float = 0.03
    brokerage_per_order: float = 20.0


class IndianPaperTrader:
    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits
        self.cash = limits.starting_cash
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []

    @staticmethod
    def sma(values: list[float], window: int) -> float | None:
        return sum(values[-window:]) / window if len(values) >= window else None

    def mark_to_market(self, latest: dict[str, float]) -> float:
        return self.cash + sum(p.quantity * latest[p.symbol] for p in self.positions.values())

    def sell(self, day: str, position: Position, price: float, reason: str) -> None:
        self.cash += position.quantity * price - self.limits.brokerage_per_order
        self.trades.append(Trade(day, position.symbol, "SELL", position.quantity, price, reason))
        del self.positions[position.symbol]

    def buy(self, day: str, symbol: str, price: float, equity: float) -> None:
        allocation = min(equity * self.limits.max_position_fraction, self.cash - self.limits.brokerage_per_order)
        quantity = int(allocation // price)
        if quantity < 1:
            return
        cost = quantity * price + self.limits.brokerage_per_order
        self.cash -= cost
        self.positions[symbol] = Position(symbol, quantity, price, day)
        self.trades.append(Trade(day, symbol, "BUY", quantity, price, "5/20 SMA bullish crossover"))

    def run(self, candles_by_symbol: dict[str, list[Candle]]) -> dict:
        symbols = sorted(candles_by_symbol)
        by_day: dict[str, dict[str, float]] = {}
        for symbol, candles in candles_by_symbol.items():
            for candle in candles:
                by_day.setdefault(candle.day, {})[symbol] = candle.close

        histories: dict[str, list[float]] = {s: [] for s in symbols}
        for day in sorted(by_day):
            closes = by_day[day]
            # First manage open positions, including a 3% protective stop.
            for symbol, position in list(self.positions.items()):
                price = closes.get(symbol)
                if price is None:
                    continue
                short = self.sma(histories[symbol], 5)
                long = self.sma(histories[symbol], 20)
                stopped = price <= position.entry_price * (1 - self.limits.stop_loss_fraction)
                bearish = short is not None and long is not None and short < long
                if stopped or bearish:
                    self.sell(day, position, price, "3% protective stop" if stopped else "5/20 SMA bearish")

            for symbol, price in closes.items():
                history = histories[symbol]
                prior_short, prior_long = self.sma(history, 5), self.sma(history, 20)
                history.append(price)
                short, long = self.sma(history, 5), self.sma(history, 20)
                crossed_up = (
                    prior_short is not None and prior_long is not None and short is not None and long is not None
                    and prior_short <= prior_long and short > long
                )
                if crossed_up and symbol not in self.positions and len(self.positions) < self.limits.max_positions:
                    self.buy(day, symbol, price, self.mark_to_market(closes))

        latest = {s: c[-1].close for s, c in candles_by_symbol.items() if c}
        return {
            "market": "India / NSE cash equities only",
            "mode": "PAPER ONLY — no broker orders are submitted",
            "starting_cash_inr": self.limits.starting_cash,
            "cash_inr": round(self.cash, 2),
            "equity_inr": round(self.mark_to_market(latest), 2),
            "open_positions": [asdict(position) for position in self.positions.values()],
            "trades": [asdict(trade) for trade in self.trades],
        }


def normalize_indian_symbol(symbol: str) -> str:
    clean = symbol.strip().upper()
    if not clean:
        raise ValueError("Empty stock symbol")
    if clean.endswith(".NS"):
        return clean
    if clean.endswith(".NSE"):
        return clean.replace(".NSE", ".NS")
    return f"{clean}.NS"


def calculate_rsi(series: pd.Series, period: int = 14) -> float:
    if len(series) < period + 1:
        return 50.0
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean().iloc[-1]
    avg_loss = loss.rolling(window=period, min_periods=period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def analyze_symbol(symbol: str, period: str = "6mo", interval: str = "1d") -> dict:
    normalized = normalize_indian_symbol(symbol)
    try:
        ticker = yf.Ticker(normalized)
        hist = ticker.history(period=period, interval=interval, auto_adjust=True, actions=False)
        if hist.empty:
            return {
                "symbol": normalized,
                "status": "NO_DATA",
                "reason": "No market data returned for this symbol.",
            }
        closes = pd.to_numeric(hist["Close"], errors="coerce").dropna()
        if closes.empty:
            return {
                "symbol": normalized,
                "status": "NO_DATA",
                "reason": "Missing close-price data.",
            }

        last_price = float(closes.iloc[-1])
        sma_20 = float(closes.tail(20).mean()) if len(closes) >= 20 else float(closes.mean())
        sma_50 = float(closes.tail(50).mean()) if len(closes) >= 50 else float(closes.mean())
        rsi = calculate_rsi(closes, 14)
        recent_change = ((last_price / closes.iloc[-2]) - 1.0) * 100 if len(closes) >= 2 else 0.0
        volume = float(hist["Volume"].iloc[-1]) if "Volume" in hist.columns else 0.0

        if last_price > sma_20 > sma_50 and 50 <= rsi <= 70:
            signal = "BUY"
        elif last_price < sma_20 < sma_50 and 30 <= rsi <= 50:
            signal = "SELL"
        else:
            signal = "HOLD"

        trend_score = 50
        if last_price > sma_20:
            trend_score += 20
        else:
            trend_score -= 20
        if sma_20 > sma_50:
            trend_score += 15
        else:
            trend_score -= 15
        trend_score += int((rsi - 50) * 0.5)
        trend_score = max(0, min(100, trend_score))

        return {
            "symbol": normalized,
            "status": "OK",
            "last_price": round(last_price, 2),
            "sma_20": round(sma_20, 2),
            "sma_50": round(sma_50, 2),
            "rsi_14": round(rsi, 2),
            "daily_change_pct": round(recent_change, 2),
            "volume": int(volume),
            "signal": signal,
            "analysis_score": trend_score,
            "trend": "bullish" if last_price > sma_20 and sma_20 > sma_50 else "bearish" if last_price < sma_20 and sma_20 < sma_50 else "neutral",
        }
    except Exception as exc:  # pragma: no cover - network or exchange errors are runtime-dependent.
        return {
            "symbol": normalized,
            "status": "ERROR",
            "reason": str(exc),
        }


def load_symbols_from_file(path: Path) -> list[str]:
    symbols: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                row = [entry.strip() for entry in line.split(",") if entry.strip()]
                symbols.extend(row)
            else:
                symbols.append(line)
    return symbols


def analyze_market(symbols: list[str], period: str = "6mo", interval: str = "1d", min_score: int | None = None, top_n: int | None = None) -> dict:
    cleaned = []
    for symbol in symbols:
        clean = symbol.strip()
        if clean:
            cleaned.append(clean)
    unique = []
    seen: set[str] = set()
    for symbol in cleaned:
        normalized = normalize_indian_symbol(symbol)
        if normalized not in seen:
            unique.append(normalized)
            seen.add(normalized)

    results = [analyze_symbol(symbol, period=period, interval=interval) for symbol in unique]
    filtered = [item for item in results if item.get("status") == "OK"]
    if min_score is not None:
        filtered = [item for item in filtered if item.get("analysis_score", 0) >= min_score]
    if top_n is not None and top_n > 0:
        filtered = sorted(filtered, key=lambda item: item.get("analysis_score", 0), reverse=True)[:top_n]
    ranked = sorted(filtered, key=lambda item: item.get("analysis_score", 0), reverse=True)
    return {
        "market": "India / NSE",
        "period": period,
        "interval": interval,
        "symbols_analyzed": len(ranked),
        "results": ranked,
    }


def load_csv(path: Path) -> list[Candle]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = csv.DictReader(file)
        candles = [Candle(row["date"], float(row["close"])) for row in rows]
    if not candles:
        raise ValueError(f"No candles in {path}")
    return sorted(candles, key=lambda candle: candle.day)


def main() -> None:
    parser = argparse.ArgumentParser(description="India-only NSE paper trading agent and market analysis tool")
    parser.add_argument("data_dir", type=Path, nargs="?", help="CSV directory: SYMBOL.csv with date,close columns")
    parser.add_argument("--output", type=Path, default=Path("paper_trading_report.json"))
    parser.add_argument("--live", action="store_true", help="Fetch near-real-time Indian market data for analysis")
    parser.add_argument("--all", action="store_true", help="Analyze the broader Indian NSE watchlist instead of a small default sample")
    parser.add_argument("--symbols", nargs="*", help="Indian stock symbols to analyze. Example: RELIANCE TCS INFY")
    parser.add_argument("--symbols-file", type=Path, help="Text/CSV file containing one Indian stock symbol per line")
    parser.add_argument("--period", default="6mo", help="History window for live data analysis")
    parser.add_argument("--interval", default="1d", help="Interval for live data analysis, such as 1d, 1h, or 5m")
    parser.add_argument("--min-score", type=int, default=None, help="Keep only stocks with analysis_score >= value")
    parser.add_argument("--top", type=int, default=None, help="Return only the top N ranked stocks")
    parser.add_argument("--csv", type=Path, default=None, help="Save filtered live-analysis results as CSV")
    args = parser.parse_args()

    if args.live or args.symbols or args.symbols_file or args.all:
        requested = list(args.symbols or [])
        if args.symbols_file is not None:
            requested.extend(load_symbols_from_file(args.symbols_file))
        if args.all:
            requested = INDIAN_STOCK_WATCHLIST
        elif not requested:
            requested = DEFAULT_INDIAN_SYMBOLS

        analysis = analyze_market(requested, period=args.period, interval=args.interval, min_score=args.min_score, top_n=args.top)
        if args.csv is not None:
            rows = analysis.get("results", [])
            df = pd.DataFrame(rows)
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.csv, index=False)
        args.output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        print(json.dumps(analysis, indent=2))
        return

    if args.data_dir is None:
        parser.error("Either provide a CSV data directory for paper trading or use --live/--symbols to analyze live Indian stocks.")

    csvs = sorted(args.data_dir.glob("*.csv"))
    if not csvs:
        raise SystemExit("No CSV files found. Add one or more SYMBOL.csv files.")
    candles = {csv_file.stem.upper(): load_csv(csv_file) for csv_file in csvs}
    report = IndianPaperTrader(RiskLimits()).run(candles)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
