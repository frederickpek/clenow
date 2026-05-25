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
    gap_window: int = 90
    gap_threshold: float | None = None
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
        gap_window=config.gap_window,
        trading_days=config.trading_days,
    )
    if summary.empty:
        return summary

    ranked = apply_filters(summary, config=config)
    ranked = ranked.sort_values("momentum_score", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def rank_momentum_asof(
    prices: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp | str,
    config: RankingConfig | None = None,
    universe: set[str] | None = None,
    include_disqualified: bool = False,
) -> pd.DataFrame:
    """Rank using only rows through as_of_date, optionally retaining ineligible rows."""
    config = config or RankingConfig()
    as_of_timestamp = pd.Timestamp(as_of_date)
    asof_prices = prices.loc[prices["date"] <= as_of_timestamp].copy()
    if universe is not None:
        asof_prices = asof_prices.loc[asof_prices["ticker"].isin(universe)]

    summary = summarize_universe(
        asof_prices,
        lookback_days=config.lookback_days,
        atr_window=config.atr_window,
        trend_ma_window=config.trend_ma_window,
        gap_window=config.gap_window,
        trading_days=config.trading_days,
    )
    if summary.empty:
        return summary

    summary = summary.sort_values("momentum_score", ascending=False).reset_index(drop=True)
    summary.insert(0, "rank", range(1, len(summary) + 1))
    summary["disqualification_reason"] = disqualification_reasons(summary, config=config)
    summary["is_eligible"] = summary["disqualification_reason"].eq("")
    if include_disqualified:
        return summary
    return summary.loc[summary["is_eligible"]].reset_index(drop=True)


def disqualification_reasons(summary: pd.DataFrame, *, config: RankingConfig) -> pd.Series:
    """Return pipe-delimited disqualification reasons for each candidate row."""
    atr_column = f"ATR{config.atr_window}"
    gap_column = f"max_gap_{config.gap_window}"
    reasons = pd.Series("", index=summary.index, dtype="object")

    def add_reason(mask: pd.Series, reason: str) -> None:
        reasons.loc[mask] = reasons.loc[mask].where(
            reasons.loc[mask].eq(""),
            reasons.loc[mask] + "|",
        ) + reason

    add_reason(summary["momentum_score"].isna(), "missing_momentum")
    add_reason(summary["r_squared"].isna(), "missing_r_squared")
    add_reason(summary[atr_column].isna() | (summary[atr_column] <= 0), "invalid_atr")
    add_reason(summary["last_price"] < config.min_price, "below_min_price")
    add_reason(summary["avg_dollar_volume_20"] < config.min_avg_dollar_volume, "below_min_liquidity")
    if config.require_positive_slope:
        add_reason(summary["annualized_slope"] <= 0, "non_positive_slope")
    if config.require_above_trend_ma:
        add_reason(~summary["above_trend_ma"], "below_trend_ma")
    if config.gap_threshold is not None and gap_column in summary:
        add_reason(summary[gap_column] >= config.gap_threshold, "large_gap")

    return reasons


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
    if config.gap_threshold is not None:
        mask &= filtered[f"max_gap_{config.gap_window}"] < config.gap_threshold

    filtered = filtered.loc[mask].copy()
    return filtered
