import unittest
import numpy as np
import pandas as pd
from datetime import datetime
from utilities import preprocess_data, add_technical_indicators
from trading_bot import train, test, params

def generate_ohlcv_series(start_price, mu, sigma, num_candles, freq='H'):
    """Generate OHLCV data using Geometric Brownian Motion."""
    prices = [start_price]
    ohlcv_data = []
    timestamps = pd.date_range(start=datetime.now(), periods=num_candles, freq=freq)

    dt = 1 / (24 if freq == 'h' else 1)

    for i in range(num_candles):
        open_price = prices[-1]
        close_price = open_price * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal())
        high_price = max(open_price, close_price) * (1 + np.random.uniform(0, 0.01))
        low_price = min(open_price, close_price) * (1 - np.random.uniform(0, 0.01))
        volume = np.random.randint(100, 1000)
        ohlcv_data.append([timestamps[i], open_price, high_price, low_price, close_price, volume])
        prices.append(close_price)
        
    return pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

class TestMarketConditions(unittest.TestCase):
    def test_all_symbols(self):
        num_candles = 250
        freq = 'h'

        market_conditions = [
            ("TUP", "Trending Up", 0.1, 0.2),
            ("TDO", "Trending Down", -0.1, 0.2),
            ("RANG", "Ranging", 0.0, 0.2),
            ("HVOL", "High Volatility", 0.0, 0.5),
            ("LVOL", "Low Volatility", 0.0, 0.1),
            ("CONS", "Consolidating Market", 0.0, 0.05),
            ("BULL", "Bullish", 0.15, 0.25),
            ("BEAR", "Bearish", -0.15, 0.25),
            ("SIDE", "Sideways", 0.0, 0.15),
            ("XBULL", "Extreme Bullish", 0.3, 0.3),
            ("XBEAR", "Extreme Bearish", -0.3, 0.3)
        ]
        
        valid_symbols = []
        dfs = []
        
        for symbol, condition, mu, sigma in market_conditions:
            start_price = np.random.randint(100, 101)  # Randomize start price for each symbol
            df = generate_ohlcv_series(start_price, mu, sigma, num_candles, freq)
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
            # Convert relevant columns to numeric types
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
            
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df.set_index('timestamp', inplace=True)
            
            df = add_technical_indicators(df)
            
            valid_symbols.append(symbol)
            dfs.append(df.dropna())

            # For this example, we'll just print the generated DataFrame
            print(f"Generated data for {symbol} under {condition}:")
            # print(df.head())
            
        data, mapping, timestamps = preprocess_data(dfs, valid_symbols)
        valid_symbols = sorted(valid_symbols)
        
        if data is not None:
        
            # Split data into training and testing sets
            split_index = int(len(data) * params['training_percentage'])
            train_data = data[:split_index]
            test_data = data[split_index:]

            # Example usage
            model, output_dir = train(train_data, mapping, valid_symbols, timestamps[:split_index], params['total_timesteps'])
            if model:
                test_performance_score = test(model, valid_symbols, test_data, mapping, timestamps[split_index:], valid_symbols, output_dir)

if __name__ == '__main__':
    unittest.main()