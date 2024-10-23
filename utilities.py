import requests
import pandas as pd
import unittest
import logging
import numpy as np
import ta
import warnings
import tensorflow as tf
from unsupervised import train_model_unsupervised, normalize_data
# Suppress specific FutureWarning from pandas
warnings.filterwarnings("ignore", category=FutureWarning, module="ta.trend")

def add_market_regime_ids(training_data, num_clusters=5, max_iterations=100):
    # Normalize training data
    normalized_training_data, scaler = normalize_data(training_data)

    # Apply K-Means clustering
    result = train_model_unsupervised(normalized_training_data, num_clusters=num_clusters, max_iterations=max_iterations)

    # Assign clusters
    cluster_assignments = tf.argmin(tf.stack([
        tf.norm(normalized_training_data - tf.reshape(centroid, (1, normalized_training_data.shape[1], normalized_training_data.shape[2])), axis=[1, 2]) 
        for centroid in result['centroids']
    ]), axis=0).numpy()

    # Create regime ID
    regime_id = cluster_assignments.reshape(-1, 1)

    # Reshape regime_id to match the dimensions of training_data
    regime_id_reshaped = np.repeat(regime_id, training_data.shape[1], axis=1)  # Repeat regime_id for each symbol
    regime_id_reshaped = regime_id_reshaped[:, :, np.newaxis]  # Add a new axis for concatenation

    # Concatenate regime_id with the original training data
    training_data_with_regime = np.concatenate((training_data, regime_id_reshaped), axis=2)

    return training_data_with_regime, scaler

def add_technical_indicators(df):
    logging.debug("Adding technical indicators")
    df['ema_short'] = ta.trend.EMAIndicator(close=df['close'], window=12).ema_indicator()
    df['ema_long'] = ta.trend.EMAIndicator(close=df['close'], window=26).ema_indicator()
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    macd = ta.trend.MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    df['boll_hband'] = ta.volatility.BollingerBands(close=df['close']).bollinger_hband()
    df['boll_mband'] = ta.volatility.BollingerBands(close=df['close']).bollinger_mavg()
    df['boll_lband'] = ta.volatility.BollingerBands(close=df['close']).bollinger_lband()
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    
    # Add DMI and ADX
    dmi = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['adx'] = dmi.adx()
    df['adx_neg'] = dmi.adx_neg()
    df['adx_pos'] = dmi.adx_pos()
    
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
    df['vwap'] = ta.volume.VolumeWeightedAveragePrice(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).volume_weighted_average_price()
    
    df['parabolic_sar'] = ta.trend.PSARIndicator(high=df['high'], low=df['low'], close=df['close']).psar()
    # ichimoku = ta.trend.IchimokuIndicator(high=df['high'], low=df['low'])
    # df['ichimoku_base_line'] = ichimoku.ichimoku_base_line()
    # df['ichimoku_conversion_line'] = ichimoku.ichimoku_conversion_line()
    # df['ichimoku_a'] = ichimoku.ichimoku_a()
    # df['ichimoku_b'] = ichimoku.ichimoku_b()
    stochastic = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
    df['stoch_k'] = stochastic.stoch()
    df['stoch_d'] = stochastic.stoch_signal()
    df['roc'] = ta.momentum.ROCIndicator(close=df['close']).roc()

    # Check for NaN values before interpolation
    if df.isnull().values.any():
        logging.debug("NaN values detected before interpolation.")
    
    # Fill NaN values using interpolation
    df.interpolate(method='linear', inplace=True)

    # Fill any remaining NaN values at the beginning with the first valid observation
    df.bfill(inplace=True)  # Backward fill to handle leading NaNs

    # Check for NaN values after filling
    if df.isnull().values.any():
        logging.warning("NaN values detected after filling.")

    logging.debug(f"Technical indicators added: {df.head()}")
    return df


def fetch_binance_klines(symbol, interval, limit):
    logging.info(f"Fetching klines for symbol: {symbol}, interval: {interval}, limit: {limit}")
    url = f"https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    response = requests.get(url, params=params)
    data = response.json()
    if not data:
        logging.warning(f"No data for symbol: {symbol}")
        return None
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'number_of_trades', 
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Convert relevant columns to numeric types
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
    
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df.set_index('timestamp', inplace=True)
    logging.debug(f"DataFrame for {symbol}:\n{df.head()}")  # Debug statement
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    logging.debug(f"Data fetched for {symbol}: {df.head()}")
    return df

def preprocess_data(dfs, symbols):
    logging.info("Starting preprocessing of data")
    
    # Filter out None dataframes and corresponding symbols
    valid_data = [(df, symbol) for df, symbol in zip(dfs, symbols) if df is not None and not df.empty]
    logging.info(f"Valid dataframes count: {len(valid_data)}")
    if not valid_data:
        raise ValueError("No valid dataframes available for preprocessing")
    
    dfs, symbols = zip(*valid_data)
    logging.info(f"Symbols with valid data: {symbols}")
    
    # Find common timestamps across all dataframes
    common_timestamps = set(dfs[0].index)
    for df in dfs[1:]:
        common_timestamps = common_timestamps.intersection(set(df.index))
    
    # Convert common_timestamps to a sorted list
    common_timestamps = sorted(list(common_timestamps))
    
    if not common_timestamps:
        logging.error("No common timestamps found across all dataframes")
        for symbol, df in zip(symbols, dfs):
            logging.info(f"Timestamp range for {symbol}: {df.index.min()} to {df.index.max()}")
        raise ValueError("No common timestamps found across all dataframes")
    
    # Filter each dataframe to include only common timestamps
    filtered_dfs = [df.loc[common_timestamps].sort_index() for df in dfs]
    
    # Combine dataframes into a single dataframe with multi-index
    combined_df = pd.concat(filtered_dfs, keys=symbols, names=['symbol', 'timestamp'])
    logging.info(f"Combined DataFrame shape: {combined_df.shape}")
    logging.info(f"Unique timestamps: {combined_df.index.get_level_values('timestamp').nunique()}")
    for symbol in symbols:
        logging.info(f"Rows for {symbol}: {combined_df.loc[symbol].shape[0]}")
    
    # Check for empty combined dataframe
    if combined_df.empty:
        raise ValueError("Combined dataframe is empty.")
    
    # Check for NaN values before reshaping
    if combined_df.isnull().values.any():
        raise ValueError("Data contains NaN values before reshaping.")
    logging.info("No NaN values detected before reshaping.")
    
    # Pivot the dataframe to get the desired shape
    matrix = combined_df.unstack(level=0).swaplevel(axis=1).sort_index(axis=1)
    logging.info(f"Dataframe pivoted to desired shape: {matrix.shape}")
    
    # Handle NaN values after pivoting
    if matrix.isnull().values.any():
        matrix = matrix.ffill().bfill()
        logging.info("NaN values handled after pivoting.")
        if matrix.isnull().values.any():
            raise ValueError("Data contains NaN values after filling.")
    
    # Reshape the matrix to have the shape (timestamps, symbols, features)
    num_timestamps = matrix.shape[0]
    num_symbols = len(symbols)
    num_features = matrix.shape[1] // num_symbols
    reshaped_matrix = matrix.values.reshape(num_timestamps, num_symbols, num_features)
    logging.info(f"Data reshaped to final format: {reshaped_matrix.shape}")
    
    # Use the new method to add market regime IDs
    training_data_with_regime, scaler = add_market_regime_ids(reshaped_matrix, num_clusters=3, max_iterations=100)
    logging.info(f"Added market regime IDs: {training_data_with_regime.shape}")
    
    # Check for NaN values after reshaping
    if np.any(np.isnan(reshaped_matrix)):
        raise ValueError("Data contains NaN values after reshaping.")
    logging.info("No NaN values detected after final reshaping.")
    
    # Extract columns from the matrix
    columns = sorted(matrix.columns.get_level_values(1).unique().tolist())
    columns.append('regime')  # Add 'regime' to the list of columns
    mapping = {col: i for i, col in enumerate(columns)}
    timestamps = matrix.index.values
    
    logging.debug(f"Data combined and reshaped: {combined_df.head()}")
    return training_data_with_regime, mapping, timestamps 

def net_profit(profits):
    # Calculate net profit from returns
    return np.sum(np.array(profits))

def sharpe_ratio(returns):
    returns = np.array(returns)
    if len(returns) < 2:
        return 0  # Not enough data to calculate Sharpe ratio
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    # Assuming risk-free rate is 0 for simplicity
    sharpe_ratio = mean_return / std_return * np.sqrt(252) if std_return != 0 else 0
    return sharpe_ratio

def max_drawdown(returns):
    returns = np.array(returns)
    if len(returns) < 2:
        return 0  # Not enough data to calculate max drawdown
    
    cumulative_returns = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative_returns)
    
    # Handle cases where peak is zero by setting drawdown to zero
    with np.errstate(divide='ignore', invalid='ignore'):
        drawdown = np.where(peak == 0, 0, (peak - cumulative_returns) / peak)
    
    # Replace NaN values with zero
    drawdown = np.nan_to_num(drawdown, nan=0.0)
    
    max_drawdown = np.max(drawdown)
    return max_drawdown

def risk_to_reward_ratio(returns):
    # Convert to numpy array if it's a list
    returns = np.array(returns)
    
    gains = returns[returns > 0]
    losses = -returns[returns < 0]  # Convert losses to positive values
    
    if len(gains) == 0 or len(losses) == 0:
        return 0  # Avoid division by zero
    
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    
    return avg_gain / avg_loss if avg_loss != 0 else 0

def calculate_leverage(df, mapping, collateral, volume_factor_base=10000):
    from trading_bot import financial_params
    params = financial_params
    volatilities = df[:, mapping['atr']]
    volumes = df[:, mapping['volume']]
    closes = df[:, mapping['close']]
    
    # Handle cases where volatility is NaN or zero
    valid_volatilities = np.where((~np.isnan(volatilities)) & (volatilities != 0), volatilities, np.inf)
    
    position_sizes = collateral / valid_volatilities
    adjusted_position_sizes = position_sizes * (volumes / volume_factor_base)
    position_values = adjusted_position_sizes * closes
    leverages = position_values / collateral
    
    # # Ensure leverage is within acceptable range
    # leverages = np.clip(leverages, LEVERAGE_MIN, LEVERAGE_MAX)
    
    # Normalize leverages
    leverage_min = np.min(leverages)
    leverage_max = np.max(leverages)
    
    if leverage_max == leverage_min:
        normalized_leverages = np.full_like(leverages, params['leverage_min'])
    else:
        normalized_leverages = params['leverage_min'] + (leverages - leverage_min) * (params['leverage_max'] - params['leverage_min']) / (leverage_max - leverage_min)
    
    # Handle NaN values
    normalized_leverages = np.where(np.isnan(normalized_leverages), params['leverage_min'], normalized_leverages)
    
    return np.round(normalized_leverages).astype(int) 

# Leverage to Liquidation Threshold mapping
LEVERAGE_LIQUIDATION_RANGES = [
    (1, 1, 100.00),
    (1, 2, 89.84),
    (2, 5, 89.60),
    (5, 10, 89.20),
    (10, 15, 88.80),
    (15, 20, 88.40),
    (20, 25, 88.00),
    (25, 30, 85.46),
    (30, 35, 82.91),
    (35, 40, 80.37),
    (40, 45, 77.83),
    (45, 50, 75.29),
    (50, 55, 72.74),
    (55, 60, 70.20),
    (60, 65, 69.80),
    (65, 70, 69.40),
    (70, 75, 69.00),
    (75, 80, 68.60),
    (80, 85, 68.20),
    (85, 90, 67.80),
    (90, 95, 67.40),
    (95, 100, 67.00),
    (100, 105, 66.60),
    (105, 110, 66.20),
    (110, 115, 65.80),
    (115, 120, 65.40),
    (120, 125, 65.00),
    (125, 130, 64.60),
    (130, 135, 64.20),
    (135, 140, 63.80),
    (140, 145, 63.40),
    (145, 151, 63.00)
]

def get_liquidation_threshold(leverage):
    for min_lev, max_lev, threshold in LEVERAGE_LIQUIDATION_RANGES:
        if min_lev <= leverage < max_lev:
            # logging.debug(f"Leverage: {leverage}, Threshold: {threshold}")
            return threshold
    return None

def compute_liquidation_price(entry_price, leverage, position_type):
    threshold = get_liquidation_threshold(leverage)
    if position_type == 'long':
        return entry_price * (1 - threshold / (leverage * 100))
    elif position_type == 'short':
        return entry_price * (1 + threshold / (leverage * 100))
    return None

# def compute_tp(entry_price, position_type, atr_value):
#     from trading_bot import ATR_TP, ATR_SL
#     if position_type == 'long':
#         tp_price = entry_price + ATR_TP * atr_value
#     elif position_type == 'short':
#         tp_price = entry_price - ATR_TP * atr_value
#     else:
#         tp_price = None
    
#     # logging.debug(f"Computed TP: {tp_price} for entry price: {entry_price}, position type: {position_type}, ATR: {atr_value}")
#     return tp_price

# def compute_sl(entry_price, position_type, atr_value):
#     from trading_bot import ATR_TP, ATR_SL
#     if position_type == 'long':
#         sl_price = entry_price - ATR_SL * atr_value
#     elif position_type == 'short':
#         sl_price = entry_price + ATR_SL * atr_value
#     else:
#         sl_price = None
    
#     # logging.debug(f"Computed SL: {sl_price} for entry price: {entry_price}, position type: {position_type}, ATR: {atr_value}")
#     return sl_price

class TestUtils(unittest.TestCase):

    def setUp(self):
        self.symbols = ['BTC', 'ETH', 'BNB', 'SOL', 'NEAR', 'FTM', 'ADA', 'LINK', 'SHIB', 'BONK']
        self.dfs = [fetch_binance_klines(symbol + 'USDT', '1h', 100) for symbol in self.symbols]

    def test_preprocess_data(self):
        matrix, _ = preprocess_data(self.dfs, self.symbols)
        self.assertEqual(matrix.shape, (100, len(self.symbols), 6))  # 100 timestamps, 2 symbols, 6 features
        
        # Check for NaN values in the matrix
        self.assertFalse(np.any(np.isnan(matrix)), "Data contains NaN values.")
    
    # def test_compute_liquidation_price(self):
    #     self.assertAlmostEqual(compute_liquidation_price(100, 10, 'long'), 91.12)
    #     self.assertAlmostEqual(compute_liquidation_price(100, 10, 'short'), 108.88)
    #     self.assertAlmostEqual(compute_liquidation_price(100, 50, 'long'), 98.5452)
    #     self.assertAlmostEqual(compute_liquidation_price(100, 50, 'short'), 101.4548)
    #     self.assertIsNone(compute_liquidation_price(100, 10, 'invalid'))

    # def test_compute_liquidation_price_gains(self):
    #     self.assertAlmostEqual(compute_liquidation_price(147.024, 10, 'long'), 135.062, delta=2)
    #     self.assertAlmostEqual(compute_liquidation_price(6.15195, 35, 'short'), 6.28742, places=1)
    #     self.assertAlmostEqual(compute_liquidation_price(63470.4, 90, 'long'), 63015.1, delta=21)
    #     self.assertIsNone(compute_liquidation_price(100, 10, 'invalid'))

    # def test_compute_tp(self):
    #     self.assertAlmostEqual(compute_tp(100, 'long', 2), 104)
    #     self.assertAlmostEqual(compute_tp(100, 'short', 2), 96)
    #     self.assertIsNone(compute_tp(100, 'invalid', 2))

    # def test_compute_sl(self):
    #     self.assertAlmostEqual(compute_sl(100, 'long', 2), 98)
    #     self.assertAlmostEqual(compute_sl(100, 'short', 2), 102)
    #     self.assertIsNone(compute_sl(100, 'invalid', 2))


        
if __name__ == '__main__':
    unittest.main()
