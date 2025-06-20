import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from data import calculate_macd, calculate_rsi,calculate_bollinger_bands

class StockDataset(Dataset):
    """Custom Dataset for stock data"""
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class StockPredictor(nn.Module):
    """Simple Neural Network for Stock Price Prediction"""
    def __init__(self, input_size, hidden_sizes=[64, 32, 16], dropout_rate=0.2):
        super(StockPredictor, self).__init__()

        layers = []
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),  # Helps with training stability
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size

        # Output layer (single value for price prediction)
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)

def prepare_data():
    """Prepare stock data for training"""
    print("📊 Loading and preparing stock data...")

    # Get more data - extend the date range significantly
    # We need at least 30+ days for meaningful training
    stock = yf.download('PLTR', start='2024-01-01', end='2025-06-13')

    print(f"📅 Downloaded {len(stock)} days of data")

    # Add technical indicators with smaller windows to preserve more data
    stock['SMA_5'] = stock['Close'].rolling(window=5).mean()  # Shorter SMA
    stock['MACD'] = calculate_macd(stock['Close'])
    stock['RSI'] = calculate_rsi(stock['Close'], window=14)
    stock['BB_High'], stock['BB_Low'] = calculate_bollinger_bands(stock['Close'], window=10, num_std=2)  # Shorter BB window

    # Create additional features
    stock['Volume_MA'] = stock['Volume'].rolling(window=3).mean()  # Shorter volume MA
    stock['Price_Change'] = stock['Close'].pct_change()
    stock['High_Low_Pct'] = (stock['High'] - stock['Low']) / stock['Close']

    # Add more simple features that don't require windows
    stock['Open_Close_Ratio'] = stock['Open'] / stock['Close']
    stock['High_Close_Ratio'] = stock['High'] / stock['Close']
    stock['Volume_Norm'] = stock['Volume'] / stock['Volume'].rolling(window=20).mean()

    # Drop rows with NaN values (caused by rolling windows)
    stock_clean = stock.dropna()

    print(f"📊 After cleaning: {len(stock_clean)} usable data points")

    # Check if we have enough data
    if len(stock_clean) < 20:
        print("⚠️ WARNING: Very little data available. Consider:")
        print("   - Extending date range further back")
        print("   - Using simpler features")
        print("   - Different stock symbol")

        # Use simpler features if we don't have enough data
        feature_columns = [
            'Open', 'High', 'Low', 'Volume',
            'Price_Change', 'High_Low_Pct', 'Open_Close_Ratio', 'High_Close_Ratio'
        ]
        print("🔄 Using simplified feature set...")
    else:
        # Use full feature set
        feature_columns = [
            'Open', 'High', 'Low', 'Volume',
            'SMA_5', 'MACD', 'RSI', 'BB_High', 'BB_Low',
            'Volume_MA', 'Price_Change', 'High_Low_Pct',
            'Open_Close_Ratio', 'High_Close_Ratio', 'Volume_Norm'
        ]

    # Target column (what we want to predict)
    target_column = 'Close'

    # Prepare features and targets
    X = stock_clean[feature_columns].values
    y = stock_clean[target_column].values.reshape(-1, 1)  # Reshape for proper dimensions

    print(f"📈 Final data shape: {X.shape}, Target shape: {y.shape}")
    print(f"📊 Features: {feature_columns}")

    return X, y, feature_columns

def create_sequences(X, y, sequence_length=5):
    """Create sequences for time series prediction (optional - makes it more sophisticated)"""
    sequences_X, sequences_y = [], []

    for i in range(len(X) - sequence_length):
        sequences_X.append(X[i:i + sequence_length])
        sequences_y.append(y[i + sequence_length])

    return np.array(sequences_X), np.array(sequences_y)

def train_model():
    """Main training function"""
    print("🚀 Starting Stock Price Prediction Training...")

    # Prepare data
    X, y, feature_columns = prepare_data()

    # Option 1: Simple approach (each day predicts next day)
    print("📝 Using simple day-to-day prediction...")

    # Normalize the features (very important for neural networks!)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # Split data (time series - don't shuffle!)
    # Ensure we have minimum training samples
    min_train_samples = 10
    if len(X_scaled) < min_train_samples + 5:  # Need at least 15 total samples
        print(f"⚠️ ERROR: Only {len(X_scaled)} samples available. Need at least {min_train_samples + 5}.")
        print("📅 Try extending the date range further back (e.g., start='2023-01-01')")
        return None, None, None, None

    split_idx = max(min_train_samples, int(len(X_scaled) * 0.8))  # At least 10 training samples

    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

    print(f"📊 Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Ensure we have both training and test data
    if len(X_train) == 0 or len(X_test) == 0:
        print("⚠️ ERROR: Insufficient data for train/test split")
        return None, None, None, None

    # Create datasets and dataloaders
    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)  # Don't shuffle time series!
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Create model
    input_size = X_scaled.shape[1]  # Number of features
    model = StockPredictor(input_size=input_size, hidden_sizes=[64, 32, 16], dropout_rate=0.3)

    print(f"🧠 Model created with input size: {input_size}")
    print(f"🏗️ Architecture: {input_size} → 64 → 32 → 16 → 1")

    # Loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Training loop
    num_epochs = 50  # Reduced for smaller dataset
    train_losses = []
    test_losses = []

    print("🏃 Starting training...")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            # Forward pass
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        # Testing phase
        model.eval()
        test_loss = 0.0
        test_batches = 0

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                test_loss += loss.item()
                test_batches += 1

        # Calculate average losses (protect against division by zero)
        avg_train_loss = train_loss / max(num_batches, 1)
        avg_test_loss = test_loss / max(test_batches, 1)

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        # Update learning rate
        scheduler.step(avg_test_loss)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {avg_train_loss:.6f}")
            print(f"  Test Loss:  {avg_test_loss:.6f}")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 40)

    # Save the model
    model_save_path = 'stock_predictor.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'feature_columns': feature_columns,
        'model_config': {
            'input_size': input_size,
            'hidden_sizes': [64, 32, 16],
            'dropout_rate': 0.3
        }
    }, model_save_path)

    print(f"💾 Model saved to {model_save_path}")

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='red')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Make predictions and plot
    model.eval()
    with torch.no_grad():
        train_predictions = model(torch.FloatTensor(X_train)).numpy()
        test_predictions = model(torch.FloatTensor(X_test)).numpy()

    # Convert back to original scale
    train_pred_original = scaler_y.inverse_transform(train_predictions)
    test_pred_original = scaler_y.inverse_transform(test_predictions)
    y_train_original = scaler_y.inverse_transform(y_train)
    y_test_original = scaler_y.inverse_transform(y_test)

    plt.subplot(1, 2, 2)
    plt.plot(range(len(y_train_original)), y_train_original, label='Actual (Train)', color='blue', alpha=0.7)
    plt.plot(range(len(y_train_original)), train_pred_original, label='Predicted (Train)', color='lightblue', alpha=0.7)
    plt.plot(range(len(y_train_original), len(y_train_original) + len(y_test_original)),
             y_test_original, label='Actual (Test)', color='red', alpha=0.7)
    plt.plot(range(len(y_train_original), len(y_train_original) + len(y_test_original)),
             test_pred_original, label='Predicted (Test)', color='pink', alpha=0.7)

    plt.title('Stock Price Predictions')
    plt.xlabel('Time')
    plt.ylabel('Stock Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Calculate and print metrics
    train_rmse = np.sqrt(np.mean((y_train_original - train_pred_original) ** 2))
    test_rmse = np.sqrt(np.mean((y_test_original - test_pred_original) ** 2))

    print("\n📊 Final Results:")
    print(f"Training RMSE: ${train_rmse:.4f}")
    print(f"Test RMSE: ${test_rmse:.4f}")
    print(f"Average stock price: ${np.mean(y):.2f}")
    print(f"Test error as % of avg price: {(test_rmse / np.mean(y)) * 100:.2f}%")

    return model, scaler_X, scaler_y, feature_columns

def load_and_predict(new_data_point=None):
    """Load saved model and make predictions"""
    print("📥 Loading saved model...")

    # Load the saved model
    checkpoint = torch.load('stock_predictor.pth')
    # Recreate the model
    model_config = checkpoint['model_config']
    model = StockPredictor(
        input_size=model_config['input_size'],
        hidden_sizes=model_config['hidden_sizes'],
        dropout_rate=model_config['dropout_rate']
    )

    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load scalers
    scaler_X = checkpoint['scaler_X']
    scaler_y = checkpoint['scaler_y']
    feature_columns = checkpoint['feature_columns']

    print("✅ Model loaded successfully!")
    print(f"📊 Features used: {feature_columns}")

    if new_data_point is not None:
        # Make prediction on new data
        new_data_scaled = scaler_X.transform(new_data_point.reshape(1, -1))

        with torch.no_grad():
            prediction_scaled = model(torch.FloatTensor(new_data_scaled))
            prediction_original = scaler_y.inverse_transform(prediction_scaled.numpy())

        print(f"🔮 Predicted stock price: ${prediction_original[0][0]:.2f}")
        return prediction_original[0][0]

    return model, scaler_X, scaler_y, feature_columns

if __name__ == "__main__":
    print("🎯 Stock Price Prediction with Neural Networks")
    print("=" * 50)

    # Train the model
    result = train_model()

    if result[0] is not None:  # Check if training was successful
        model, scaler_X, scaler_y, feature_columns = result

        print("\n" + "=" * 50)
        print("🎉 Training completed!")
        print("💡 You can now use load_and_predict() to make new predictions")
        print("📁 Model saved as 'stock_predictor.pth'")
    else:
        print("\n" + "=" * 50)
        print("❌ Training failed due to insufficient data")
        print("🔧 Try:")
        print("   - Extending date range: start='2023-01-01' or earlier")
        print("   - Using a different stock ticker")
        print("   - Collecting more recent data")