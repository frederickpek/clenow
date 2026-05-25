"""ATR-based position sizing and cash-constrained buy-list construction."""

from __future__ import annotations

from collections.abc import Mapping
from math import floor

import pandas as pd


def size_position(
    *,
    account_value: float,
    atr: float,
    daily_move_target: float = 0.001,
) -> int:
    """Calculate target shares using AccountValue * daily_move_target / ATR."""
    if account_value <= 0:
        raise ValueError("account_value must be positive.")
    if atr <= 0:
        return 0
    return floor(account_value * daily_move_target / atr)


def build_buy_list(
    ranked: pd.DataFrame,
    *,
    account_value: float,
    available_cash: float,
    daily_move_target: float = 0.001,
    atr_column: str = "ATR20",
    existing_positions: Mapping[str, int] | None = None,
    allow_partial_final_position: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk down ranked candidates and buy target shares until cash is exhausted."""
    if account_value <= 0:
        raise ValueError("account_value must be positive.")
    if available_cash < 0:
        raise ValueError("available_cash cannot be negative.")
    required_columns = {"ticker", "last_price", atr_column}
    missing = required_columns.difference(ranked.columns)
    if missing:
        raise ValueError(f"Missing required ranking columns: {sorted(missing)}")

    existing_positions = existing_positions or {}
    remaining_cash = float(available_cash)
    buys: list[dict[str, float | int | str]] = []
    diagnostics: list[dict[str, float | int | str]] = []

    for _, row in ranked.iterrows():
        ticker = str(row["ticker"])
        last_price = float(row["last_price"])
        atr = float(row[atr_column])
        target_shares = size_position(
            account_value=account_value,
            atr=atr,
            daily_move_target=daily_move_target,
        )
        current_shares = int(existing_positions.get(ticker, 0))
        shares_to_buy = max(target_shares - current_shares, 0)
        target_cost = shares_to_buy * last_price

        if shares_to_buy == 0:
            diagnostics.append(_diagnostic(row, target_shares, 0, 0.0, remaining_cash, "no_buy_needed"))
            continue

        if target_cost > remaining_cash:
            if allow_partial_final_position:
                affordable_shares = floor(remaining_cash / last_price)
                if affordable_shares > 0:
                    cost = affordable_shares * last_price
                    remaining_cash -= cost
                    buys.append(_buy_row(row, target_shares, affordable_shares, cost, remaining_cash))
                    diagnostics.append(
                        _diagnostic(
                            row,
                            target_shares,
                            affordable_shares,
                            cost,
                            remaining_cash,
                            "partial_buy",
                        )
                    )
                    break

            diagnostics.append(
                _diagnostic(row, target_shares, shares_to_buy, target_cost, remaining_cash, "insufficient_cash")
            )
            continue

        remaining_cash -= target_cost
        buys.append(_buy_row(row, target_shares, shares_to_buy, target_cost, remaining_cash))
        diagnostics.append(
            _diagnostic(row, target_shares, shares_to_buy, target_cost, remaining_cash, "buy")
        )

    return pd.DataFrame(buys), pd.DataFrame(diagnostics)


def _buy_row(
    row: pd.Series,
    target_shares: int,
    shares_to_buy: int,
    estimated_cost: float,
    remaining_cash: float,
) -> dict[str, float | int | str]:
    output = row.to_dict()
    output.update(
        {
            "target_shares": target_shares,
            "shares_to_buy": shares_to_buy,
            "estimated_cost": estimated_cost,
            "remaining_cash": remaining_cash,
        }
    )
    return output


def _diagnostic(
    row: pd.Series,
    target_shares: int,
    shares_to_buy: int,
    estimated_cost: float,
    remaining_cash: float,
    action: str,
) -> dict[str, float | int | str]:
    return {
        "ticker": row["ticker"],
        "rank": row.get("rank", ""),
        "target_shares": target_shares,
        "shares_to_buy": shares_to_buy,
        "estimated_cost": estimated_cost,
        "remaining_cash": remaining_cash,
        "action": action,
    }
