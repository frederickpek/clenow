"""Weekly Clenow-style backtest engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import floor

import pandas as pd

from clenow.portfolio import size_position
from clenow.ranking import RankingConfig, rank_momentum_asof


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for the weekly Wednesday momentum backtest."""

    initial_cash: float = 100_000.0
    weeks: int = 104
    trade_weekday: int = 2
    index_proxy: str = "SPY"
    index_trend_ma_window: int = 100
    top_fraction: float = 0.20
    daily_move_target: float = 0.001
    slippage_bps: float = 5.0
    gap_threshold: float = 0.15
    risk_rebalance_every_n_trades: int = 2
    risk_rebalance_threshold: float = 0.20
    allow_partial_final_position: bool = False
    show_progress: bool = False
    progress_every: int = 1
    ranking: RankingConfig = field(
        default_factory=lambda: RankingConfig(
            lookback_days=90,
            atr_window=20,
            trend_ma_window=100,
            gap_window=90,
            gap_threshold=None,
            require_positive_slope=True,
            require_above_trend_ma=False,
        )
    )


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    holdings: pd.DataFrame
    rebalance_log: pd.DataFrame


def run_backtest(
    prices: pd.DataFrame,
    *,
    config: BacktestConfig | None = None,
    universe: set[str] | None = None,
) -> BacktestResult:
    """Run the weekly strategy backtest on tidy OHLCV prices."""
    config = config or BacktestConfig()
    prices = _prepare_prices(prices)
    universe = universe or set(prices["ticker"].unique()).difference({config.index_proxy})

    trade_dates = _trade_dates(prices, config=config)
    total_trade_dates = len(trade_dates)
    holdings: dict[str, int] = {}
    cash = float(config.initial_cash)
    trade_rows: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []
    holding_rows: list[dict[str, object]] = []
    rebalance_rows: list[dict[str, object]] = []

    for cycle, trade_date in enumerate(trade_dates, start=1):
        signal_date = _previous_trading_date(prices, trade_date)
        if signal_date is None:
            _print_progress(
                config,
                cycle=cycle,
                total=total_trade_dates,
                trade_date=trade_date,
                message="skipped: no prior signal date",
            )
            continue

        ranking = rank_momentum_asof(
            prices,
            as_of_date=signal_date,
            config=config.ranking,
            universe=universe,
            include_disqualified=True,
        )
        if ranking.empty:
            _print_progress(
                config,
                cycle=cycle,
                total=total_trade_dates,
                trade_date=trade_date,
                message="skipped: no ranking candidates",
            )
            continue

        ranking = _add_backtest_eligibility(ranking, config=config)
        ranking_by_ticker = ranking.set_index("ticker", drop=False)
        top_cutoff = max(1, floor(len(ranking) * config.top_fraction))
        index_positive = _index_in_positive_trend(prices, signal_date, config=config)

        cash += _sell_disqualified_positions(
            prices,
            holdings,
            ranking_by_ticker,
            trade_date=trade_date,
            signal_date=signal_date,
            cash=cash,
            config=config,
            top_cutoff=top_cutoff,
            trade_rows=trade_rows,
        )

        account_value = _account_value(prices, holdings, cash, trade_date, signal_date)
        if index_positive:
            cash = _buy_replacements(
                prices,
                holdings,
                ranking,
                trade_date=trade_date,
                signal_date=signal_date,
                cash=cash,
                account_value=account_value,
                config=config,
                top_cutoff=top_cutoff,
                trade_rows=trade_rows,
            )

        account_value = _account_value(prices, holdings, cash, trade_date, signal_date)
        if config.risk_rebalance_every_n_trades > 0 and cycle % config.risk_rebalance_every_n_trades == 0:
            cash = _rebalance_position_risk(
                prices,
                holdings,
                ranking_by_ticker,
                trade_date=trade_date,
                signal_date=signal_date,
                cash=cash,
                account_value=account_value,
                config=config,
                trade_rows=trade_rows,
                rebalance_rows=rebalance_rows,
            )

        account_value = _account_value(prices, holdings, cash, trade_date, signal_date)
        equity_rows.append(
            {
                "date": trade_date,
                "signal_date": signal_date,
                "cash": cash,
                "holdings_value": account_value - cash,
                "equity": account_value,
                "index_positive_trend": index_positive,
                "positions": len(holdings),
                "top_cutoff": top_cutoff,
            }
        )
        holding_rows.extend(
            _snapshot_holdings(
                prices,
                holdings,
                ranking_by_ticker,
                trade_date=trade_date,
                signal_date=signal_date,
                account_value=account_value,
            )
        )
        _print_progress(
            config,
            cycle=cycle,
            total=total_trade_dates,
            trade_date=trade_date,
            message=(
                f"equity=${account_value:,.2f}, cash=${cash:,.2f}, "
                f"positions={len(holdings)}, trades={len(trade_rows)}"
            ),
        )

    return BacktestResult(
        equity_curve=pd.DataFrame(equity_rows),
        trades=pd.DataFrame(trade_rows),
        holdings=pd.DataFrame(holding_rows),
        rebalance_log=pd.DataFrame(rebalance_rows),
    )


def _print_progress(
    config: BacktestConfig,
    *,
    cycle: int,
    total: int,
    trade_date: pd.Timestamp,
    message: str,
) -> None:
    if not config.show_progress:
        return
    progress_every = max(config.progress_every, 1)
    if cycle % progress_every != 0 and cycle != total:
        return
    print(f"[{cycle:>4}/{total}] {trade_date.date()} - {message}", flush=True)


def _prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"date", "ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"}
    missing = required_columns.difference(prices.columns)
    if missing:
        raise ValueError(f"Missing required price columns: {sorted(missing)}")

    prepared = prices.copy()
    prepared["date"] = pd.to_datetime(prepared["date"]).dt.tz_localize(None)
    return prepared.sort_values(["ticker", "date"]).reset_index(drop=True)


def _trade_dates(prices: pd.DataFrame, *, config: BacktestConfig) -> list[pd.Timestamp]:
    dates = (
        prices.loc[prices["ticker"].ne(config.index_proxy), "date"]
        .drop_duplicates()
        .sort_values()
    )
    dates = dates.loc[dates.dt.weekday == config.trade_weekday]
    if config.weeks > 0:
        dates = dates.tail(config.weeks)
    return list(dates)


def _previous_trading_date(prices: pd.DataFrame, trade_date: pd.Timestamp) -> pd.Timestamp | None:
    dates = prices.loc[prices["date"] < trade_date, "date"].drop_duplicates().sort_values()
    if dates.empty:
        return None
    return dates.iloc[-1]


def _add_backtest_eligibility(ranking: pd.DataFrame, *, config: BacktestConfig) -> pd.DataFrame:
    ranking = ranking.copy()
    gap_column = f"max_gap_{config.ranking.gap_window}"
    ranking["large_gap"] = ranking[gap_column] >= config.gap_threshold
    ranking["backtest_disqualification_reason"] = ranking["disqualification_reason"]
    ranking.loc[~ranking["above_trend_ma"], "backtest_disqualification_reason"] += "|below_trend_ma"
    ranking.loc[ranking["large_gap"], "backtest_disqualification_reason"] += "|large_gap"
    ranking["backtest_disqualification_reason"] = ranking[
        "backtest_disqualification_reason"
    ].str.strip("|")
    ranking["backtest_eligible"] = ranking["backtest_disqualification_reason"].eq("")
    return ranking


def _index_in_positive_trend(
    prices: pd.DataFrame,
    signal_date: pd.Timestamp,
    *,
    config: BacktestConfig,
) -> bool:
    index_prices = prices.loc[
        (prices["ticker"] == config.index_proxy) & (prices["date"] <= signal_date)
    ].sort_values("date")
    if len(index_prices) < config.index_trend_ma_window:
        raise ValueError(
            f"Not enough {config.index_proxy} history to calculate "
            f"MA{config.index_trend_ma_window} on {signal_date.date()}."
        )
    trend_ma = index_prices["Adj Close"].rolling(config.index_trend_ma_window).mean().iloc[-1]
    return bool(index_prices["Adj Close"].iloc[-1] > trend_ma)


def _sell_disqualified_positions(
    prices: pd.DataFrame,
    holdings: dict[str, int],
    ranking_by_ticker: pd.DataFrame,
    *,
    trade_date: pd.Timestamp,
    signal_date: pd.Timestamp,
    cash: float,
    config: BacktestConfig,
    top_cutoff: int,
    trade_rows: list[dict[str, object]],
) -> float:
    proceeds = 0.0
    for ticker, shares in list(holdings.items()):
        reason = _sell_reason(ticker, ranking_by_ticker, top_cutoff=top_cutoff)
        if reason is None:
            continue

        execution_price = _execution_price(prices, ticker, trade_date, signal_date)
        trade_value = shares * execution_price
        slippage = _slippage(trade_value, config=config)
        proceeds += trade_value - slippage
        trade_rows.append(
            _trade_row(
                date=trade_date,
                ticker=ticker,
                action="sell",
                shares=shares,
                price=execution_price,
                slippage=slippage,
                reason=reason,
                cash_after=cash + proceeds,
            )
        )
        del holdings[ticker]
    return proceeds


def _sell_reason(ticker: str, ranking_by_ticker: pd.DataFrame, *, top_cutoff: int) -> str | None:
    if ticker not in ranking_by_ticker.index:
        return "missing_or_left_universe"
    row = ranking_by_ticker.loc[ticker]
    reasons = []
    if int(row["rank"]) > top_cutoff:
        reasons.append("outside_top_20_pct")
    if not bool(row["above_trend_ma"]):
        reasons.append("below_trend_ma")
    if bool(row["large_gap"]):
        reasons.append("large_gap")
    if not bool(row["is_eligible"]):
        reasons.append(str(row["disqualification_reason"]))
    return "|".join(reason for reason in reasons if reason) or None


def _buy_replacements(
    prices: pd.DataFrame,
    holdings: dict[str, int],
    ranking: pd.DataFrame,
    *,
    trade_date: pd.Timestamp,
    signal_date: pd.Timestamp,
    cash: float,
    account_value: float,
    config: BacktestConfig,
    top_cutoff: int,
    trade_rows: list[dict[str, object]],
) -> float:
    atr_column = f"ATR{config.ranking.atr_window}"
    for _, row in ranking.iterrows():
        if int(row["rank"]) > top_cutoff:
            break
        ticker = str(row["ticker"])
        if ticker in holdings or not bool(row["backtest_eligible"]):
            continue

        execution_price = _execution_price(prices, ticker, trade_date, signal_date)
        target_shares = size_position(
            account_value=account_value,
            atr=float(row[atr_column]),
            daily_move_target=config.daily_move_target,
        )
        if target_shares <= 0:
            continue

        total_cost = _buy_cost(target_shares, execution_price, config=config)
        shares = target_shares
        if total_cost > cash:
            if not config.allow_partial_final_position:
                continue
            shares = floor(cash / (execution_price * (1 + config.slippage_bps / 10_000)))
            total_cost = _buy_cost(shares, execution_price, config=config)
            if shares <= 0:
                continue

        slippage = _slippage(shares * execution_price, config=config)
        cash -= total_cost
        holdings[ticker] = shares
        trade_rows.append(
            _trade_row(
                date=trade_date,
                ticker=ticker,
                action="buy",
                shares=shares,
                price=execution_price,
                slippage=slippage,
                reason="ranked_buy",
                cash_after=cash,
            )
        )
    return cash


def _rebalance_position_risk(
    prices: pd.DataFrame,
    holdings: dict[str, int],
    ranking_by_ticker: pd.DataFrame,
    *,
    trade_date: pd.Timestamp,
    signal_date: pd.Timestamp,
    cash: float,
    account_value: float,
    config: BacktestConfig,
    trade_rows: list[dict[str, object]],
    rebalance_rows: list[dict[str, object]],
) -> float:
    atr_column = f"ATR{config.ranking.atr_window}"
    for ticker, current_shares in list(holdings.items()):
        if ticker not in ranking_by_ticker.index:
            continue
        row = ranking_by_ticker.loc[ticker]
        target_shares = size_position(
            account_value=account_value,
            atr=float(row[atr_column]),
            daily_move_target=config.daily_move_target,
        )
        if target_shares <= 0:
            continue

        current_risk = current_shares * float(row[atr_column])
        target_risk = target_shares * float(row[atr_column])
        risk_deviation = abs(current_risk - target_risk) / target_risk
        rebalance_rows.append(
            {
                "date": trade_date,
                "ticker": ticker,
                "current_shares": current_shares,
                "target_shares": target_shares,
                "risk_deviation": risk_deviation,
            }
        )
        if risk_deviation < config.risk_rebalance_threshold:
            continue

        share_delta = target_shares - current_shares
        execution_price = _execution_price(prices, ticker, trade_date, signal_date)
        if share_delta > 0:
            total_cost = _buy_cost(share_delta, execution_price, config=config)
            if total_cost > cash:
                continue
            slippage = _slippage(share_delta * execution_price, config=config)
            cash -= total_cost
            holdings[ticker] = target_shares
            action = "rebalance_buy"
        else:
            sell_shares = abs(share_delta)
            trade_value = sell_shares * execution_price
            slippage = _slippage(trade_value, config=config)
            cash += trade_value - slippage
            holdings[ticker] = target_shares
            action = "rebalance_sell"

        trade_rows.append(
            _trade_row(
                date=trade_date,
                ticker=ticker,
                action=action,
                shares=abs(share_delta),
                price=execution_price,
                slippage=slippage,
                reason="risk_rebalance",
                cash_after=cash,
            )
        )
        if holdings[ticker] == 0:
            del holdings[ticker]
    return cash


def _execution_price(
    prices: pd.DataFrame,
    ticker: str,
    trade_date: pd.Timestamp,
    signal_date: pd.Timestamp,
) -> float:
    execution_row = prices.loc[(prices["ticker"] == ticker) & (prices["date"] == trade_date)]
    if not execution_row.empty and pd.notna(execution_row["Open"].iloc[-1]):
        return float(execution_row["Open"].iloc[-1])

    fallback = prices.loc[(prices["ticker"] == ticker) & (prices["date"] <= signal_date)].sort_values("date")
    if fallback.empty:
        raise ValueError(f"No execution or fallback price found for {ticker} on {trade_date.date()}.")
    return float(fallback["Close"].iloc[-1])


def _account_value(
    prices: pd.DataFrame,
    holdings: dict[str, int],
    cash: float,
    trade_date: pd.Timestamp,
    signal_date: pd.Timestamp,
) -> float:
    holdings_value = sum(
        shares * _execution_price(prices, ticker, trade_date, signal_date)
        for ticker, shares in holdings.items()
    )
    return cash + holdings_value


def _snapshot_holdings(
    prices: pd.DataFrame,
    holdings: dict[str, int],
    ranking_by_ticker: pd.DataFrame,
    *,
    trade_date: pd.Timestamp,
    signal_date: pd.Timestamp,
    account_value: float,
) -> list[dict[str, object]]:
    rows = []
    for ticker, shares in holdings.items():
        price = _execution_price(prices, ticker, trade_date, signal_date)
        ranking_row = ranking_by_ticker.loc[ticker] if ticker in ranking_by_ticker.index else {}
        atr = float(ranking_row.get("ATR20", 0.0)) if hasattr(ranking_row, "get") else 0.0
        rows.append(
            {
                "date": trade_date,
                "ticker": ticker,
                "shares": shares,
                "price": price,
                "market_value": shares * price,
                "weight": (shares * price) / account_value if account_value > 0 else 0.0,
                "rank": ranking_row.get("rank", "") if hasattr(ranking_row, "get") else "",
                "atr_risk": shares * atr,
            }
        )
    return rows


def _buy_cost(shares: int, price: float, *, config: BacktestConfig) -> float:
    trade_value = shares * price
    return trade_value + _slippage(trade_value, config=config)


def _slippage(trade_value: float, *, config: BacktestConfig) -> float:
    return trade_value * config.slippage_bps / 10_000


def _trade_row(
    *,
    date: pd.Timestamp,
    ticker: str,
    action: str,
    shares: int,
    price: float,
    slippage: float,
    reason: str,
    cash_after: float,
) -> dict[str, object]:
    return {
        "date": date,
        "ticker": ticker,
        "action": action,
        "shares": shares,
        "price": price,
        "trade_value": shares * price,
        "slippage": slippage,
        "reason": reason,
        "cash_after": cash_after,
    }
