"""Momentum and risk indicators used by the Clenow notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import linregress


def true_range(prices: pd.DataFrame) -> pd.Series:
    """Calculate true range for one ticker's OHLC data."""
    previous_close = prices["Close"].shift(1)
    ranges = pd.concat(
        [
            prices["High"] - prices["Low"],
            (prices["High"] - previous_close).abs(),
            (prices["Low"] - previous_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def add_atr(prices: pd.DataFrame, *, window: int = 20) -> pd.DataFrame:
    """Add a rolling ATR column to tidy OHLCV prices."""
    prices = prices.sort_values(["ticker", "date"]).copy()
    atr_parts = []
    for _, group in prices.groupby("ticker", sort=False):
        group_atr = true_range(group).rolling(window=window, min_periods=window).mean()
        atr_parts.append(group_atr)
    prices[f"ATR{window}"] = pd.concat(atr_parts).sort_index()
    return prices


def candle_gap(prices: pd.DataFrame) -> pd.Series:
    """Calculate the absolute open-to-close move for each daily candle."""
    return (prices["Open"] / prices["Close"] - 1).abs()


def add_candle_gap(prices: pd.DataFrame) -> pd.DataFrame:
    """Add a single-candle open/close gap column."""
    prices = prices.copy()
    prices["candle_gap"] = candle_gap(prices)
    return prices


def exponential_regression_momentum(
    prices: pd.Series,
    *,
    trading_days: int = 252,
) -> tuple[float, float, float]:
    """Return annualized log-price slope, R-squared, and adjusted slope score."""
    clean_prices = prices.dropna()
    clean_prices = clean_prices[clean_prices > 0]
    if len(clean_prices) < 2:
        return np.nan, np.nan, np.nan

    x = np.arange(len(clean_prices))
    y = np.log(clean_prices.to_numpy(dtype=float))
    regression = linregress(x, y)
    annualized_slope = np.expm1(regression.slope * trading_days)
    r_squared = regression.rvalue**2
    score = annualized_slope * r_squared
    return annualized_slope, r_squared, score


def summarize_ticker(
    prices: pd.DataFrame,
    *,
    lookback_days: int = 90,
    atr_window: int = 20,
    trend_ma_window: int = 100,
    gap_window: int = 90,
    trading_days: int = 252,
) -> dict[str, float | str | pd.Timestamp | bool]:
    """Summarize one ticker with the latest price, ATR, trend, and momentum score."""
    prices = prices.sort_values("date").copy()
    prices[f"ATR{atr_window}"] = true_range(prices).rolling(
        window=atr_window,
        min_periods=atr_window,
    ).mean()
    prices[f"MA{trend_ma_window}"] = prices["Adj Close"].rolling(
        window=trend_ma_window,
        min_periods=trend_ma_window,
    ).mean()
    prices["avg_dollar_volume_20"] = (
        prices["Adj Close"] * prices["Volume"]
    ).rolling(window=20, min_periods=20).mean()
    prices["candle_gap"] = candle_gap(prices)
    prices[f"max_gap_{gap_window}"] = prices["candle_gap"].rolling(
        window=gap_window,
        min_periods=1,
    ).max()

    recent = prices.tail(lookback_days)
    slope, r_squared, score = exponential_regression_momentum(
        recent["Adj Close"],
        trading_days=trading_days,
    )
    latest = prices.iloc[-1]
    return {
        "ticker": str(latest["ticker"]),
        "date": latest["date"],
        "last_price": float(latest["Adj Close"]),
        f"ATR{atr_window}": float(latest[f"ATR{atr_window}"]),
        f"MA{trend_ma_window}": float(latest[f"MA{trend_ma_window}"]),
        "avg_dollar_volume_20": float(latest["avg_dollar_volume_20"]),
        "candle_gap": float(latest["candle_gap"]),
        f"max_gap_{gap_window}": float(latest[f"max_gap_{gap_window}"]),
        "annualized_slope": slope,
        "r_squared": r_squared,
        "momentum_score": score,
        "above_trend_ma": bool(latest["Adj Close"] > latest[f"MA{trend_ma_window}"]),
        "history_days": len(prices),
    }


def summarize_universe(
    prices: pd.DataFrame,
    *,
    lookback_days: int = 90,
    atr_window: int = 20,
    trend_ma_window: int = 100,
    gap_window: int = 90,
    trading_days: int = 252,
) -> pd.DataFrame:
    """Create one latest-row indicator summary per ticker."""
    summaries = []
    required_columns = {"date", "ticker", "High", "Low", "Close", "Adj Close", "Volume"}
    missing = required_columns.difference(prices.columns)
    if missing:
        raise ValueError(f"Missing required price columns: {sorted(missing)}")

    for _, group in prices.groupby("ticker", sort=True):
        if len(group) < max(lookback_days, atr_window, trend_ma_window):
            continue
        summaries.append(
            summarize_ticker(
                group,
                lookback_days=lookback_days,
                atr_window=atr_window,
                trend_ma_window=trend_ma_window,
                gap_window=gap_window,
                trading_days=trading_days,
            )
        )
    return pd.DataFrame(summaries)
