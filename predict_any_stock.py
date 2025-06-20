"""
Universal Stock Predictor - Works with any ticker!
Usage: python predict_any_stock.py TSLA
       python predict_any_stock.py AAPL --timeframe 5day
       python predict_any_stock.py NVDA --train
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import argparse
import sys
import os
from model import StockDataset, StockPredictor

class UniversalStockPredictor:
    """Stock predictor that works with any ticker symbol"""

    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.feature_columns = [
            'Return_1d', 'Return_3d', 'Return_5d', 'Return_10d',
            'Position_5d', 'Position_20d',
            'Volume_Ratio_5d', 'Volume_Ratio_20d',
            'Price_vs_MA5', 'Price_vs_MA20',
            'Volatility_5d', 'Volatility_20d',
            'Intraday_Range', 'Open_Gap', 'RSI'
        ]

    def get_model_filename(self, timeframe='1day'):
        """Generate filename for specific ticker and timeframe"""
        return f'model_{self.ticker}_{timeframe}.pth'

    def calculate_features(self, stock_data):
        """Calculate relative features for any stock"""
        stock = stock_data.copy()

        # Returns
        stock['Return_1d'] = stock['Close'].pct_change(1)
        stock['Return_3d'] = stock['Close'].pct_change(3)
        stock['Return_5d'] = stock['Close'].pct_change(5)
        stock['Return_10d'] = stock['Close'].pct_change(10)

        # Position in range
        stock['Position_5d'] = (stock['Close'] - stock['Close'].rolling(5).min()) / (stock['Close'].rolling(5).max() - stock['Close'].rolling(5).min() + 1e-8)
        stock['Position_20d'] = (stock['Close'] - stock['Close'].rolling(20).min()) / (stock['Close'].rolling(20).max() - stock['Close'].rolling(20).min() + 1e-8)

        # Volume ratios
        stock['Volume_Ratio_5d'] = stock['Volume'] / stock['Volume'].rolling(5).mean()
        stock['Volume_Ratio_20d'] = stock['Volume'] / stock['Volume'].rolling(20).mean()

        # Price vs moving averages
        stock['Price_vs_MA5'] = (stock['Close'] - stock['Close'].rolling(5).mean()) / stock['Close'].rolling(5).mean()
        stock['Price_vs_MA20'] = (stock['Close'] - stock['Close'].rolling(20).mean()) / stock['Close'].rolling(20).mean()

        # Volatility
        stock['Volatility_5d'] = stock['Return_1d'].rolling(5).std()
        stock['Volatility_20d'] = stock['Return_1d'].rolling(20).std()

        # Intraday features
        stock['Intraday_Range'] = (stock['High'] - stock['Low']) / stock['Open']
        stock['Open_Gap'] = (stock['Open'] - stock['Close'].shift(1)) / stock['Close'].shift(1)

        # RSI
        stock['RSI'] = self.calculate_rsi(stock['Close'])

        return stock

    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        avg_gain = gains.ewm(span=window).mean()
        avg_loss = losses.ewm(span=window).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        return 100 - (100 / (1 + rs))

    def download_data(self, period='2y'):
        """Download stock data"""
        try:
            print(f"📊 Downloading {self.ticker} data...")
            stock = yf.download(self.ticker, period=period)

            if len(stock) == 0:
                raise ValueError(f"No data found for ticker {self.ticker}")

            print(f"✅ Downloaded {len(stock)} days of data")
            return stock

        except Exception as e:
            print(f"❌ Error downloading {self.ticker}: {e}")
            print("💡 Make sure the ticker symbol is correct")
            return None

    def train_model(self, timeframe='1day'):
        """Train model for specific ticker and timeframe"""

        print(f"🎯 TRAINING {self.ticker} MODEL - {timeframe.upper()}")
        print("="*50)

        # Download data
        stock = self.download_data()
        if stock is None:
            return None

        # Calculate features
        stock = self.calculate_features(stock)

        # Define target based on timeframe
        timeframe_days = {
            '1day': 1, '3day': 3, '5day': 5,
            '1week': 5, '2week': 10, '1month': 20
        }

        days = timeframe_days.get(timeframe, 1)

        if days == 1:
            stock['Target'] = stock['Close'].pct_change().shift(-days)
        else:
            stock['Target'] = (stock['Close'].shift(-days) / stock['Close']) - 1

        # Clean data
        stock_clean = stock.dropna()
        stock_clean = stock_clean[:-days]  # Remove last N days

        if len(stock_clean) < 100:
            print(f"❌ Not enough data for {self.ticker} (only {len(stock_clean)} samples)")
            return None

        print(f"📊 Using {len(stock_clean)} samples for training")

        # Prepare data
        X = stock_clean[self.feature_columns].values
        y = stock_clean['Target'].values.reshape(-1, 1)

        # Split and scale
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)

        # Create and train model
        model = StockPredictor(input_size=len(self.feature_columns), hidden_sizes=[64, 32, 16], dropout_rate=0.3)

        train_dataset = StockDataset(X_train_scaled, y_train_scaled)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print("🏃 Training model...")

        for epoch in range(50):
            model.train()
            total_loss = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(train_loader)
                print(f"  Epoch {epoch+1}: Loss = {avg_loss:.6f}")

        # Test model
        model.eval()
        with torch.no_grad():
            test_pred_scaled = model(torch.FloatTensor(X_test_scaled))
            test_pred = scaler_y.inverse_transform(test_pred_scaled.numpy())

        # Calculate metrics
        rmse = np.sqrt(np.mean((test_pred - y_test) ** 2))
        direction_acc = np.mean(np.sign(test_pred.flatten()) == np.sign(y_test.flatten())) * 100

        print(f"📊 RMSE: {rmse*100:.2f}%")
        print(f"🎯 Direction Accuracy: {direction_acc:.1f}%")

        # Save model
        model_file = self.get_model_filename(timeframe)
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'feature_columns': self.feature_columns,
            'model_config': {
                'input_size': len(self.feature_columns),
                'hidden_sizes': [64, 32, 16],
                'dropout_rate': 0.3
            },
            'ticker': self.ticker,
            'timeframe': timeframe,
            'days': days,
            'training_samples': len(stock_clean),
            'performance': {
                'rmse': rmse,
                'direction_accuracy': direction_acc
            }
        }, model_file)

        print(f"💾 Model saved: {model_file}")
        return model_file

    def predict(self, timeframe='1day'):
        """Make prediction using trained model"""

        print(f"🔮 PREDICTING {self.ticker} - {timeframe.upper()}")
        print("="*40)

        model_file = self.get_model_filename(timeframe)

        # Check if model exists
        if not os.path.exists(model_file):
            print(f"❌ No model found for {self.ticker} {timeframe}")
            print(f"💡 Train first with: --train --timeframe {timeframe}")
            return None

        # Load model
        checkpoint = torch.load(model_file, weights_only=False)

        model = StockPredictor(
            input_size=checkpoint['model_config']['input_size'],
            hidden_sizes=checkpoint['model_config']['hidden_sizes'],
            dropout_rate=checkpoint['model_config']['dropout_rate']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        scaler_X = checkpoint['scaler_X']
        scaler_y = checkpoint['scaler_y']

        print(f"✅ Loaded model (Accuracy: {checkpoint['performance']['direction_accuracy']:.1f}%)")

        # Get current data
        stock = self.download_data(period='6mo')
        if stock is None:
            return None

        # Calculate features
        stock = self.calculate_features(stock)
        stock_clean = stock.dropna()

        latest_data = stock_clean.iloc[-1]
        current_price = latest_data['Close'].iloc[0] if hasattr(latest_data['Close'], 'iloc') else float(latest_data['Close'])

        print(f"📅 Date: {stock_clean.index[-1].strftime('%Y-%m-%d')}")
        print(f"💰 Current {self.ticker} price: ${current_price:.2f}")

        # Extract features
        features = []
        for feature_name in self.feature_columns:
            value = latest_data[feature_name].iloc[0] if hasattr(latest_data[feature_name], 'iloc') else float(latest_data[feature_name])
            features.append(value)

        # Make prediction
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler_X.transform(features_array)

        with torch.no_grad():
            pred_scaled = model(torch.FloatTensor(features_scaled))
            predicted_return = scaler_y.inverse_transform(pred_scaled.numpy())[0][0]

        predicted_price = current_price * (1 + predicted_return)

        # Display results
        print(f"\n🎯 {timeframe.upper()} PREDICTION:")
        print(f"   Expected return: {predicted_return*100:+.2f}%")
        print(f"   Target price: ${predicted_price:.2f}")
        print(f"   Price change: ${predicted_price - current_price:+.2f}")

        if abs(predicted_return) > 0.02:
            confidence = "🔥 Strong"
        elif abs(predicted_return) > 0.01:
            confidence = "⚡ Medium"
        else:
            confidence = "💤 Weak"

        direction = "📈 BULLISH" if predicted_return > 0 else "📉 BEARISH"
        print(f"   Signal: {direction} ({confidence})")

        return {
            'ticker': self.ticker,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'predicted_return': predicted_return,
            'timeframe': timeframe
        }

def main():
    parser = argparse.ArgumentParser(description='Universal Stock Predictor')
    parser.add_argument('ticker', help='Stock ticker symbol (e.g., TSLA, AAPL, NVDA)')
    parser.add_argument('--timeframe', '-t', default='1day',
                       choices=['1day', '3day', '5day', '1week', '2week', '1month'],
                       help='Prediction timeframe (default: 1day)')
    parser.add_argument('--train', action='store_true',
                       help='Train new model for this ticker')
    parser.add_argument('--train-all', action='store_true',
                       help='Train models for all timeframes')
    parser.add_argument('--all-timeframes', action='store_true',
                       help='Predict all timeframes')

    args = parser.parse_args()

    # Create predictor
    predictor = UniversalStockPredictor(args.ticker)

    if args.train_all:
        # Train all timeframes
        timeframes = ['1day', '3day', '5day', '1week', '2week']
        print(f"🚀 TRAINING ALL TIMEFRAMES FOR {args.ticker}")
        print("="*50)

        for tf in timeframes:
            print(f"\n📚 Training {tf} model...")
            result = predictor.train_model(tf)
            if result:
                print(f"✅ {tf} model complete")
            else:
                print(f"❌ {tf} model failed")

        print(f"\n🎉 All training complete! Now try:")
        print(f"python {sys.argv[0]} {args.ticker} --all-timeframes")

    elif args.train:
        # Train model
        result = predictor.train_model(args.timeframe)
        if result:
            print(f"\n✅ Training complete! Now predict with:")
            print(f"python {sys.argv[0]} {args.ticker} --timeframe {args.timeframe}")

    elif args.all_timeframes:
        # Predict all available timeframes
        timeframes = ['1day', '3day', '5day', '1week', '2week']
        results = {}

        print(f"🎯 ALL TIMEFRAME PREDICTIONS FOR {args.ticker}")
        print("="*50)

        for tf in timeframes:
            result = predictor.predict(tf)
            if result:
                results[tf] = result
                print()

        # Summary table
        if results:
            print("\n📊 SUMMARY TABLE")
            print("-" * 50)
            print(f"{'Timeframe':<10} {'Return':<8} {'Target Price':<12} {'Signal'}")
            print("-" * 50)

            for tf, result in results.items():
                return_pct = f"{result['predicted_return']*100:+.1f}%"
                price = f"${result['predicted_price']:.2f}"
                signal = "📈" if result['predicted_return'] > 0 else "📉"
                print(f"{tf:<10} {return_pct:<8} {price:<12} {signal}")

    else:
        # Single prediction
        predictor.predict(args.timeframe)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("🚀 UNIVERSAL STOCK PREDICTOR")
        print("="*40)
        print("Usage examples:")
        print("  python predict_any_stock.py TSLA")
        print("  python predict_any_stock.py AAPL --timeframe 5day")
        print("  python predict_any_stock.py NVDA --train")
        print("  python predict_any_stock.py MSFT --all-timeframes")
        print("\nAvailable timeframes: 1day, 3day, 5day, 1week, 2week, 1month")
        sys.exit(1)

    main()