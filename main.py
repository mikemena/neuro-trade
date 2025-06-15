import yfinance as yf
import pandas as pd
import numpy as np

# Download stock data
stock = yf.download('PLTR', start='2025-05-15', end='2025-06-13')  # Extended range
print(stock)

# Add technical indicators
stock['SMA_10'] = stock['Close'].rolling(window=10).mean()

# MACD calculation
def calculate_macd(close, fast=12, slow=26, signal=9):
    # Calculate EMAs
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    # MACD Line
    macd_line = ema_fast - ema_slow
    return macd_line

# RSI calculation
def calculate_rsi(close, window=14):
    delta = close.diff()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    avg_gain = gains.ewm(span=window, adjust=False).mean()
    avg_loss = losses.ewm(span=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    return rsi

# Add MACD
stock['MACD'] = calculate_macd(stock['Close'])
stock['RSI'] = calculate_rsi(stock['Close'], window=14)

# Print the last few rows
print(stock[['Close', 'SMA_10', 'MACD','RSI']].tail())