# Clenow Momentum Notebooks

Jupyter notebooks for a weekly S&P 500 momentum workflow inspired by Andreas F. Clenow's
`Stocks on the Move`.

The project downloads S&P 500 data, ranks stocks with a Clenow-style adjusted slope score,
and creates ATR-based buy targets for a portfolio.

## Setup

```bash
python -m virtualenv .venv
source .venv/bin/activate
python -m pip install -e .
python -m ipykernel install --user --name clenow --display-name "Python (clenow)"
jupyter lab
```

## Weekly Workflow

Run the notebooks in order:

1. `notebooks/01_download_sp500_data.ipynb`
  - Fetches current S&P 500 constituents from Wikipedia.
  - Downloads daily OHLCV history from Yahoo Finance via `yfinance`.
  - Saves cached data to `data/processed/sp500_prices.parquet`.
2. `notebooks/02_rank_momentum.ipynb`
  - Computes the adjusted momentum score:

```text
momentum_score = annualized_exponential_regression_slope * r_squared
```

- Uses adjusted close prices for the regression.
- Computes ATR20 from daily OHLC prices.
- Applies configurable price, liquidity, positive-slope, and trend filters.
- Saves rankings to `data/processed/momentum_rankings.parquet` and `.csv`.

1. `notebooks/03_portfolio_targets.ipynb`
  - Edit `ACCOUNT_VALUE`, `AVAILABLE_CASH`, and optional `EXISTING_POSITIONS`.
  - Calculates target shares with:

```text
shares = AccountValue * 0.001 / ATR20
```

- Walks down the ranked list and adds buys while enough cash remains.
- Saves `data/processed/buy_list.csv` and `data/processed/buy_list_diagnostics.csv`.

## Key Parameters

Ranking defaults live in `notebooks/02_rank_momentum.ipynb` via `RankingConfig`:

- `lookback_days=90`
- `atr_window=20`
- `trend_ma_window=100`
- `min_price=5.0`
- `min_avg_dollar_volume=10_000_000.0`
- `require_positive_slope=True`
- `require_above_trend_ma=True`

Portfolio sizing defaults live in `notebooks/03_portfolio_targets.ipynb`:

- `DAILY_MOVE_TARGET = 0.001`, which means a one-ATR daily move is targeted at 10 basis
points of account value.
- `ALLOW_PARTIAL_FINAL_POSITION = False`, so the notebook skips candidates when it cannot
afford the full suggested share count.

## Notes

This is research tooling, not financial advice. Yahoo Finance data can be revised, delayed,
or unavailable for some tickers, so inspect the diagnostics before trading.