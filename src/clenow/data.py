"""Data download and caching helpers for the notebook workflow."""

from __future__ import annotations

from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf

SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def to_yahoo_symbol(symbol: str) -> str:
    """Convert index constituent symbols to Yahoo Finance symbols."""
    return symbol.strip().replace(".", "-")


def fetch_sp500_constituents() -> pd.DataFrame:
    """Fetch the current S&P 500 constituents from Wikipedia."""
    request = Request(
        SP500_WIKIPEDIA_URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
            )
        },
    )
    with urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8")

    tables = pd.read_html(StringIO(html))
    constituents = tables[0].copy()
    constituents = constituents.rename(
        columns={
            "Symbol": "ticker",
            "Security": "name",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "industry",
        }
    )
    constituents["ticker"] = constituents["ticker"].astype(str)
    constituents["yahoo_ticker"] = constituents["ticker"].map(to_yahoo_symbol)
    return constituents[["ticker", "yahoo_ticker", "name", "sector", "industry"]]


def get_sp500_tickers() -> list[str]:
    """Return current S&P 500 tickers formatted for Yahoo Finance."""
    return fetch_sp500_constituents()["yahoo_ticker"].tolist()


def with_extra_tickers(tickers: Iterable[str], extra_tickers: Iterable[str]) -> list[str]:
    """Return Yahoo-formatted tickers plus any additional symbols, deduplicated."""
    return sorted({to_yahoo_symbol(ticker) for ticker in [*tickers, *extra_tickers]})


def download_ohlcv(
    tickers: Iterable[str],
    *,
    period: str = "3y",
    interval: str = "1d",
    start: str | None = None,
    end: str | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance and return a tidy DataFrame."""
    ticker_list = sorted({to_yahoo_symbol(ticker) for ticker in tickers})
    if not ticker_list:
        raise ValueError("At least one ticker is required.")

    raw = yf.download(
        ticker_list,
        period=None if start else period,
        interval=interval,
        start=start,
        end=end,
        auto_adjust=False,
        actions=False,
        group_by="ticker",
        progress=progress,
        threads=True,
    )
    if raw.empty:
        raise ValueError("No price data was downloaded.")

    return _yfinance_to_tidy(raw, ticker_list)


def _yfinance_to_tidy(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        if set(raw.columns.get_level_values(0)).intersection(tickers):
            stacked = raw.stack(level=0, future_stack=True)
            stacked.index = stacked.index.set_names(["date", "ticker"])
        else:
            stacked = raw.stack(level=1, future_stack=True)
            stacked.index = stacked.index.set_names(["date", "ticker"])
        tidy = stacked.reset_index()
    else:
        tidy = raw.reset_index()
        tidy["ticker"] = tickers[0]
        tidy = tidy.rename(columns={"Date": "date"})

    tidy = tidy.rename(columns={"Date": "date"})
    tidy["date"] = pd.to_datetime(tidy["date"]).dt.tz_localize(None)

    available_price_columns = [column for column in OHLCV_COLUMNS if column in tidy.columns]
    tidy = tidy[["date", "ticker", *available_price_columns]]
    tidy = tidy.dropna(subset=["Adj Close", "High", "Low", "Close"], how="any")
    tidy = tidy.sort_values(["ticker", "date"]).reset_index(drop=True)
    return tidy


def save_prices(prices: pd.DataFrame, path: str | Path) -> Path:
    """Save tidy OHLCV prices to parquet or CSV based on the file extension."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".csv":
        prices.to_csv(output_path, index=False)
    else:
        prices.to_parquet(output_path, index=False)
    return output_path


def load_prices(path: str | Path) -> pd.DataFrame:
    """Load cached tidy OHLCV prices from parquet or CSV."""
    input_path = Path(path)
    if input_path.suffix == ".csv":
        prices = pd.read_csv(input_path, parse_dates=["date"])
    else:
        prices = pd.read_parquet(input_path)
        prices["date"] = pd.to_datetime(prices["date"])
    return prices.sort_values(["ticker", "date"]).reset_index(drop=True)
