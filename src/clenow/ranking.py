"""Ranking logic for Clenow-style momentum candidates."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from clenow.indicators import summarize_universe


@dataclass(frozen=True)
class RankingConfig:
    lookback_days: int = 90
    atr_window: int = 20
    trend_ma_window: int = 100
    trading_days: int = 252
    require_positive_slope: bool = True
    require_above_trend_ma: bool = True
    min_price: float = 5.0
    min_avg_dollar_volume: float = 10_000_000.0


def rank_momentum(
    prices: pd.DataFrame,
    *,
    config: RankingConfig | None = None,
) -> pd.DataFrame:
    """Rank a price universe by annualized exponential regression slope times R-squared."""
    config = config or RankingConfig()
    summary = summarize_universe(
        prices,
        lookback_days=config.lookback_days,
        atr_window=config.atr_window,
        trend_ma_window=config.trend_ma_window,
        trading_days=config.trading_days,
    )
    if summary.empty:
        return summary

    ranked = apply_filters(summary, config=config)
    ranked = ranked.sort_values("momentum_score", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def apply_filters(summary: pd.DataFrame, *, config: RankingConfig) -> pd.DataFrame:
    """Apply configurable validity, trend, price, and liquidity filters."""
    atr_column = f"ATR{config.atr_window}"
    filtered = summary.copy()
    mask = (
        filtered["momentum_score"].notna()
        & filtered["r_squared"].notna()
        & filtered[atr_column].notna()
        & (filtered[atr_column] > 0)
        & (filtered["last_price"] >= config.min_price)
        & (filtered["avg_dollar_volume_20"] >= config.min_avg_dollar_volume)
    )
    if config.require_positive_slope:
        mask &= filtered["annualized_slope"] > 0
    if config.require_above_trend_ma:
        mask &= filtered["above_trend_ma"]

    filtered = filtered.loc[mask].copy()
    return filtered
