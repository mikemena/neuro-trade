# neuro-trade

## Create virtual env

`python -m venv neuro`

## Start the virtual env

`source neuro/bin/activate`

## Stop the virtual env

`deactivate`

## Input

### Price-Based Metrics

- Closing Price: The stock’s price at market close, often the primary target for prediction.
- Opening Price: The price at market open, useful for intraday trends.
- High/Low Prices: Daily high and low prices, capturing volatility.
- Adjusted Closing Price: Accounts for dividends and splits, ensuring consistency.
- Volume-Based Metrics
- Trading Volume: Number of shares traded, indicating market interest and liquidity.
- Volume Weighted Average Price (VWAP): Average price weighted by volume, useful for intraday trading signals.

### Technical Indicators

These are derived from price and volume data to identify trends and momentum:

- Moving Averages (MA):
  1- Simple Moving Average (SMA): Average price over a period (e.g., 10, 50, 200 days).
  2- Exponential Moving Average (EMA): Weighted average giving more importance to recent prices.
- Relative Strength Index (RSI): Measures momentum (0–100 scale) to identify overbought/oversold conditions.
- Moving Average Convergence Divergence (MACD): Difference between short-term and long-term EMAs, signaling trend changes.
- Bollinger Bands: Measure volatility by plotting bands around a moving average (±2 standard deviations).
- Stochastic Oscillator: Compares closing price to price range, indicating momentum.
- Average True Range (ATR): Measures volatility based on price range.

### Market and Sentiment Metrics

- Market Index Prices: Prices of indices like S&P 500 or NASDAQ, as stocks often correlate with the broader market.
- Volatility Index (VIX): Measures market fear/uncertainty, impacting stock volatility.
- Sentiment Data (optional): News sentiment or social media activity (e.g., X posts), though harder to quantify without advanced tools.

### Time-Based Features

- Lagged Prices/Returns: Past prices or returns (e.g., 1-day, 5-day, 10-day lags) to capture temporal dependencies.
- Day of Week/Month: Seasonal patterns in stock behavior.

## Nueral Network

- Long Short-Term Memory (LSTM) neural network
- Simple Feed-Forward (Dense) Model

- Input layer with your selected features
- 1-2 LSTM layers (50-100 neurons each)
- Dropout layers (0.2-0.3) to prevent overfitting
- Dense layer for final prediction
- Output layer (1 neuron for single price prediction)

## Usage

## Train New Model

🏃 Train New Models
`python predict_any_stock.py TSLA --train`
`python predict_any_stock.py TSLA --train --timeframe 1day`
`python predict_any_stock.py TSLA --train --timeframe 3day`
`python predict_any_stock.py TSLA --train --timeframe 5day`
`python predict_any_stock.py TSLA --train --timeframe 1week`
`python predict_any_stock.py TSLA --train --timeframe 2week`
`python predict_any_stock.py TSLA --train-all`

### Run a Prediction

🎯 Basic Prediction (1-day default)

`python predict_any_stock.py TSLA`
`python predict_any_stock.py TSLA --all-timeframes`

📅 Specific Timeframes

`python predict_any_stock.py TSLA --timeframe 5day`
`python predict_any_stock.py AAPL --timeframe 1week`
`python predict_any_stock.py NVDA --timeframe 1month`
