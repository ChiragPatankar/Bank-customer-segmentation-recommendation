import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import alpaca_trade_api as tradeapi
import time
import os
from datetime import datetime, timedelta


# Custom technical indicators (replacing TA-Lib)
def calculate_sma(data, period=30):
    return data.rolling(window=period).mean()


def calculate_ema(data, period=30):
    return data.ewm(span=period, adjust=False).mean()


def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_parabolic_sar(high, low, af_start=0.02, af_increment=0.02, af_max=0.2):
    """Simplified Parabolic SAR implementation"""
    length = len(high)
    sar = np.zeros(length)
    trend = np.zeros(length)
    ep = np.zeros(length)
    af = np.zeros(length)

    # Initialize
    trend[0] = 1  # Start with uptrend
    sar[0] = low[0]
    ep[0] = high[0]
    af[0] = af_start

    for i in range(1, length):
        # Previous trend was up
        if trend[i - 1] == 1:
            sar[i] = sar[i - 1] + af[i - 1] * (ep[i - 1] - sar[i - 1])
            # Ensure SAR is below the previous two lows
            if i >= 2:
                sar[i] = min(sar[i], low[i - 1], low[i - 2])

            # Check for trend reversal
            if low[i] < sar[i]:
                trend[i] = -1  # Downtrend
                sar[i] = ep[i - 1]
                ep[i] = low[i]
                af[i] = af_start
            else:
                trend[i] = 1  # Still uptrend
                if high[i] > ep[i - 1]:
                    ep[i] = high[i]
                    af[i] = min(af[i - 1] + af_increment, af_max)
                else:
                    ep[i] = ep[i - 1]
                    af[i] = af[i - 1]
        # Previous trend was down
        else:
            sar[i] = sar[i - 1] + af[i - 1] * (ep[i - 1] - sar[i - 1])
            # Ensure SAR is above the previous two highs
            if i >= 2:
                sar[i] = max(sar[i], high[i - 1], high[i - 2])

            # Check for trend reversal
            if high[i] > sar[i]:
                trend[i] = 1  # Uptrend
                sar[i] = ep[i - 1]
                ep[i] = high[i]
                af[i] = af_start
            else:
                trend[i] = -1  # Still downtrend
                if low[i] < ep[i - 1]:
                    ep[i] = low[i]
                    af[i] = min(af[i - 1] + af_increment, af_max)
                else:
                    ep[i] = ep[i - 1]
                    af[i] = af[i - 1]

    return sar


# Function to fetch historical data (replacing fixed CSV)
def get_historical_data(symbol, timeframe='1D', limit=1000):
    """Fetch historical data using Alpaca API"""
    try:
        # For paper trading or if using API directly:
        bars = api.get_bars(symbol, timeframe, limit=limit).df
        # Reset index to make Date a column and then set it as index again (to match CSV format)
        bars = bars.reset_index()
        bars = bars.rename(columns={"timestamp": "Date"})
        bars = bars.set_index('Date')
        return bars
    except Exception as e:
        print(f"Error fetching data: {e}")
        # Fallback to CSV if API fails
        if os.path.exists(f'{symbol}.csv'):
            print(f"Using local {symbol}.csv file as fallback")
            return pd.read_csv(f'{symbol}.csv', parse_dates=['Date'], index_col='Date')
        else:
            raise Exception(f"No data available for {symbol}")


# Function to safely execute orders with position checking
def execute_trade(symbol, side, quantity=1):
    try:
        # Check current positions
        positions = api.list_positions()
        current_position = next((p for p in positions if p.symbol == symbol), None)

        # Check if market is open
        clock = api.get_clock()
        if not clock.is_open:
            print(f"Market is closed. Current time: {clock.timestamp}")
            return False

        if side == 'buy':
            # Check if we already have a position
            if current_position:
                print(f"Already holding {current_position.qty} shares of {symbol}")
                return False

            # Check account balance
            account = api.get_account()
            if float(account.buying_power) < quantity * get_current_price(symbol):
                print("Insufficient buying power")
                return False

            api.submit_order(
                symbol=symbol,
                qty=quantity,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            print(f"Buy order executed: {quantity} shares of {symbol}")
            return True

        elif side == 'sell':
            # Check if we have a position to sell
            if not current_position:
                print(f"No position in {symbol} to sell")
                return False

            sell_qty = min(int(current_position.qty), quantity)
            api.submit_order(
                symbol=symbol,
                qty=sell_qty,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
            print(f"Sell order executed: {sell_qty} shares of {symbol}")
            return True

    except Exception as e:
        print(f"Order execution error: {e}")
        return False


# Function to get current price
def get_current_price(symbol):
    try:
        # Get the latest trade
        trades = api.get_latest_trade(symbol)
        return float(trades.price)
    except Exception as e:
        print(f"Error getting current price: {e}")
        # Fallback to last close price from bars
        try:
            bars = api.get_bars(symbol, '1D', limit=1).df
            return bars['close'].iloc[-1]
        except:
            print("Cannot get price data")
            return None


# Main trading script
def main():
    # Configuration
    symbol = 'AAPL'
    model_retrain_days = 7  # Retrain model weekly
    stop_loss = 0.02  # 2% stop loss
    target_profit = 0.05  # 5% target profit
    check_interval = 60  # Seconds between checks
    last_train_time = None
    max_iterations = 100  # Safety to prevent infinite loop

    # Use proper credentials
    api_key = os.environ.get('ALPACA_API_KEY', '')
    api_secret = os.environ.get('ALPACA_API_SECRET', '')
    base_url = os.environ.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

    if not api_key or not api_secret:
        print("API credentials not found. Please set ALPACA_API_KEY and ALPACA_API_SECRET environment variables.")
        return

    print(f"Trading algorithm for {symbol} starting...")

    # Initialize lists for visualization
    dates = []
    prices = []
    buy_signals = []
    sell_signals = []

    iteration = 0

    try:
        while iteration < max_iterations:
            iteration += 1
            current_time = datetime.now()

            # Check if we need to train/retrain the model
            if last_train_time is None or (current_time - last_train_time).days >= model_retrain_days:
                print("Training/retraining model...")

                # Get historical data
                data = get_historical_data(symbol)

                # Calculate technical indicators
                data['SMA'] = calculate_sma(data['close'])
                data['EMA'] = calculate_ema(data['close'])
                data['RSI'] = calculate_rsi(data['close'])
                data['SAR'] = calculate_parabolic_sar(data['high'].values, data['low'].values)

                # Clean data and prepare target
                data.dropna(inplace=True)
                data['Target'] = data['close'].shift(-1)
                data.dropna(inplace=True)

                # Prepare data for neural network
                features = data[['SMA', 'EMA', 'RSI', 'SAR']]
                target = data['Target']

                X_train, X_test, y_train, y_test = train_test_split(
                    features, target, test_size=0.2, random_state=42
                )

                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Build and train model
                model = keras.Sequential([
                    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1)
                ])

                model.compile(optimizer='adam', loss='mean_squared_error')

                # Use early stopping to prevent overfitting
                early_stop = keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=20, restore_best_weights=True
                )

                model.fit(
                    X_train_scaled, y_train,
                    epochs=100,  # Reduced from 500
                    batch_size=16,
                    validation_split=0.1,
                    callbacks=[early_stop],
                    verbose=1
                )

                # Evaluate model
                test_loss = model.evaluate(X_test_scaled, y_test, verbose=0)
                print(f"Model test loss: {test_loss}")

                last_train_time = current_time

            # Get latest data for prediction
            latest_data = get_historical_data(symbol, limit=50)

            # Calculate indicators
            latest_data['SMA'] = calculate_sma(latest_data['close'])
            latest_data['EMA'] = calculate_ema(latest_data['close'])
            latest_data['RSI'] = calculate_rsi(latest_data['close'])
            latest_data['SAR'] = calculate_parabolic_sar(latest_data['high'].values, latest_data['low'].values)

            # Clean data
            latest_data.dropna(inplace=True)

            if len(latest_data) == 0:
                print("No valid data available for prediction")
                time.sleep(check_interval)
                continue

            # Get features for prediction
            features = latest_data[['SMA', 'EMA', 'RSI', 'SAR']].iloc[-1].values.reshape(1, -1)
            features_scaled = scaler.transform(features)

            # Make prediction
            predicted_price = model.predict(features_scaled)[0][0]
            current_price = get_current_price(symbol)

            if current_price is None:
                print("Could not get current price, skipping trading decision")
                time.sleep(check_interval)
                continue

            # Store data for visualization
            current_date = datetime.now()
            dates.append(current_date)
            prices.append(current_price)

            # Make trading decision
            print(f"\nCurrent price: ${current_price:.2f}")
            print(f"Predicted price: ${predicted_price:.2f}")

            if predicted_price > current_price * (1 + target_profit):
                print(f"BUY signal: Current=${current_price:.2f}, Predicted=${predicted_price:.2f}")
                if execute_trade(symbol, 'buy'):
                    buy_signals.append((current_date, current_price))
                else:
                    buy_signals.append(None)
                sell_signals.append(None)
            elif predicted_price < current_price * (1 - stop_loss):
                print(f"SELL signal: Current=${current_price:.2f}, Predicted=${predicted_price:.2f}")
                if execute_trade(symbol, 'sell'):
                    sell_signals.append((current_date, current_price))
                else:
                    sell_signals.append(None)
                buy_signals.append(None)
            else:
                print("No trade signal")
                buy_signals.append(None)
                sell_signals.append(None)

            # Only plot every 5 iterations to avoid blocking execution
            if iteration % 5 == 0:
                plt.figure(figsize=(14, 7))

                # Plot price
                valid_dates = [d for d in dates if d is not None]
                valid_prices = [p for p in prices if p is not None]
                if valid_dates and valid_prices:
                    plt.plot(valid_dates, valid_prices, label='Price', alpha=0.7)

                # Plot signals
                valid_buy_signals = [(d, p) for d, p in buy_signals if d is not None and p is not None]
                valid_sell_signals = [(d, p) for d, p in sell_signals if d is not None and p is not None]

                if valid_buy_signals:
                    buy_x, buy_y = zip(*valid_buy_signals)
                    plt.scatter(buy_x, buy_y, color='green', marker='^', s=100, label='Buy')

                if valid_sell_signals:
                    sell_x, sell_y = zip(*valid_sell_signals)
                    plt.scatter(sell_x, sell_y, color='red', marker='v', s=100, label='Sell')

                plt.axhline(y=predicted_price, color='orange', linestyle='--',
                            label=f'Predicted: ${predicted_price:.2f}')
                plt.title(f'{symbol} Trading Signals')
                plt.xlabel('Date')
                plt.ylabel('Price ($)')
                plt.legend()
                plt.grid(True, alpha=0.3)

                # Save plot instead of displaying (to avoid blocking)
                plt.savefig(f'{symbol}_trading_signals.png')
                plt.close()
                print(f"Updated plot saved to {symbol}_trading_signals.png")

            # Wait before next check
            print(f"Waiting {check_interval} seconds until next check...")
            time.sleep(check_interval)

    except KeyboardInterrupt:
        print("Trading algorithm stopped by user")
    except Exception as e:
        print(f"Error in trading algorithm: {e}")
    finally:
        print("Trading algorithm terminated")


if __name__ == "__main__":
    # Initialize Alpaca API
    api_key = os.environ.get('ALPACA_API_KEY', '')
    api_secret = os.environ.get('ALPACA_API_SECRET', '')
    base_url = os.environ.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

    if not api_key or not api_secret:
        print("Please set your Alpaca API credentials as environment variables:")
        print("export ALPACA_API_KEY='your_api_key'")
        print("export ALPACA_API_SECRET='your_api_secret'")
    else:
        api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        main()