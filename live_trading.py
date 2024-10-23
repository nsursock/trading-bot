import os
import logging
from websocket import WebSocketApp
import json
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import argparse
from datetime import datetime
import pandas as pd
import pytz
import glob
from utilities import add_technical_indicators, preprocess_data, fetch_binance_klines
from interactions import open_trade, close_trade, fetch_open_trades, fetch_symbols, close_all_open_trades

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize a buffer to store messages
message_buffer = {}

# Define WebSocket event handlers
# Define WebSocket event handlers
def on_message(ws, message):
    logging.debug(f"Received raw message: {message}")  # Debug level log for raw message

    try:
        message = json.loads(message)
        logging.debug(f"Message decoded successfully")  # Info level log for successful decoding

        if 'k' in message:
            logging.debug("Key 'k' found in message")  # Debug level log for key presence
        
            candle = message['k']
            symbol = candle['s'][:-4]  # remove USDT from symbol
            logging.debug(f"Processing symbol: {symbol}")  # Info level log for symbol processing
            
            if candle['x']:
                logging.info(f"Processing closed candle for symbol {symbol}")  # Debug level log for closed candle processing
                timestamp = candle['t']
                if timestamp not in message_buffer:
                    message_buffer[timestamp] = []
                    logging.debug(f"Created new entry in message_buffer for timestamp: {timestamp}")  # Debug level log for buffer entry creation
                
                kline_data = {
                    'symbol': symbol,
                    'timestamp': candle['t'],
                    'open': float(candle['o']),
                    'high': float(candle['h']),
                    'low': float(candle['l']),
                    'close': float(candle['c']),
                    'volume': float(candle['v']),
                }
                logging.debug(f"Kline data prepared: {kline_data}")  # Info level log for kline data preparation
                
                kline_df = pd.DataFrame([kline_data])
                kline_df['timestamp'] = pd.to_datetime(kline_df['timestamp'], unit='ms')
                kline_df[['open', 'high', 'low', 'close', 'volume']] = kline_df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
                
                message_buffer[timestamp].append(kline_df)
                logging.debug(f"Appended kline data to message_buffer for timestamp: {timestamp}")  # Debug level log for data appending
                
                if len(message_buffer[timestamp]) == len(financial_params['symbols']):
                    logging.info('----------------------------------------------------------------------')
                    logging.info('----------------------------------------------------------------------')
                    logging.info(f"Received all messages for timestamp: {timestamp}")  # Info level log for all messages received
                    
                    message_buffer[timestamp] = sorted(message_buffer[timestamp], key=lambda x: x.iloc[0]['symbol'])
                    logging.info(f"Sorted message_buffer for timestamp: {timestamp}")  # Debug level log for buffer sorting
                    
                    for i in range(len(current_window)):
                        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                        
                        # Check if it's the first batch of messages
                        if len(message_buffer) == 1:
                            dfw = current_window[i]  # First batch
                        else:
                            dfw = current_window[i].iloc[1:]  # Subsequent batches
                        
                        dfc = message_buffer[timestamp][i]
                        
                        dfw.reset_index(inplace=True)
                        dfc.reset_index(inplace=True)
                        
                        # Concatenate without dropping duplicates immediately
                        current_window[i] = pd.concat([dfw[cols], dfc[cols]], axis=0)
                        
                        logging.debug(f"Before drop duplicates for symbol {sorted(financial_params['symbols'])[i]}: {len(current_window[i])} candles")  # Debug level log for window update
                        
                        # Sort and drop duplicates after concatenation
                        current_window[i] = current_window[i].sort_values(by='timestamp').drop_duplicates(subset=['timestamp'], keep='last')
                        current_window[i].set_index('timestamp', inplace=True)
                        
                        # Add technical indicators
                        current_window[i] = add_technical_indicators(current_window[i])
    
                        logging.info(f"Updated current window for symbol {sorted(financial_params['symbols'])[i]}: {len(current_window[i])} candles")  # Debug level log for window update
                        logging.debug(f"{current_window[i].head()}")  # Debug level log for window update
                    
                    # message_buffer.pop(timestamp)
                    # logging.info(f"Cleared message buffer for timestamp: {timestamp}")  # Info level log for buffer clearing
        
                    data, mapping, timestamps = preprocess_data(current_window, valid_symbols)
                    logging.info(f"Preprocessed data for current_window")  # Debug level log for data preprocessing
                    
                    live_env.envs[0].update_data(data, timestamps)
                    logging.info(f"Updated environment with new data")  # Info level log for environment update
                    
                    obs = live_env.envs[0].get_current_observation()
                    logging.info(f"Obtained current observation from environment")  # Debug level log for observation retrieval
                    
                    # action, _states = model.predict(obs)
                    # logging.info(f"Predicted action: {action}")  # Info level log for action prediction
                    
                    actions = [model.predict(obs)[0] for model in models]
                    averaged_action = sum(actions) / len(actions)
                    logging.info(f"Averaged action: {averaged_action}")  # Log the averaged action
                    
                    _, _, _, _, infos = live_env.envs[0].step(averaged_action)
                    logging.info(f"Stepped environment with action {averaged_action}")  # Debug level log for environment stepping
                    
                    # After the environment step
                    for symbol, info in infos.items():
                        logging.info('----------------------------------------------------------------------')
                        current_trades = fetch_open_trades(symbol)
                        trade_type = info.get('type')

                        # Fetch the latest price for the symbol
                        data1s = fetch_binance_klines(symbol + 'USDT', financial_params['interval'], 100)  # Assuming fetch_binance_klines returns the latest kline data
                        latest_price = data1s.iloc[-1]['close']  # Assuming the data is returned as a DataFrame

                        if trade_type in ['open_long', 'open_short']:
                            if current_trades:  # Close the latest trade if it exists
                                latest_trade = current_trades[0]
                                symbol, trade_index, is_long = latest_trade
                                close_trade(trade_index, latest_price, is_long)
                            
                            open_trade(latest_price, fetch_symbols(), symbol, trade_type, info.get('collateral'), info.get('leverage'), info.get('tp_price'), info.get('sl_price'))
                            logging.info(f"Opened {trade_type} for {symbol} with latest price {latest_price} and info {info}")
                            
                        elif trade_type == 'close' and current_trades:  # Close trades if they exist
                            latest_trade = current_trades[0]
                            symbol, trade_index, is_long = latest_trade
                            close_trade(trade_index, latest_price, is_long)
                            logging.info(f"Closed {trade_type} for {symbol} with info {info}")
                        else:
                            logging.info(f"No action required for {symbol} with type {trade_type}")

                        logging.info(f"Executed trade for {symbol} with type {trade_type} and info {info}")
                        
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {e}")
    except KeyError as e:
        logging.error(f"Key error: {e}")
    except Exception as e:
        logging.error(f"Error in on_message: {e}")
        raise e

def on_error(ws, error):
    logging.error(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    close_all_open_trades()
    logging.info(f"WebSocket closed: {close_status_code} - {close_msg}")

def on_open(ws):
    global live_env, models, financial_params, current_window, valid_symbols
    
    logging.info("WebSocket connection opened")
    logging.debug("Initializing trading bot components")

    from environment import CryptoTradingEnv
    from trading_bot import prepare, log_parameters
    from parameters import create_financial_params, selected_params
    
    financial_params = create_financial_params(selected_params)
    # financial_params['interval'] = '1m'  # for debugging
    # financial_params['cooldown_period'] = 2
    financial_params['risk_per_trade'] = 0.018
    financial_params['initial_balance'] = 5000
    log_parameters(financial_params)
    logging.debug(f"Financial parameters set: {financial_params}")

    # Prepare historical data
    reshaped_data, mapping, valid_symbols, timestamps, current_window = prepare(financial_params)
    
    if current_window is None:
        logging.error("Failed to prepare historical data. Exiting.")
        raise RuntimeError('Failed to prepare historical data. Exiting.')
    
    logging.info(f"Prepared historical data for trading with {len(valid_symbols)} symbols")
    
    params = {
        "method": "SUBSCRIBE",
        "params": [f"{symbol.lower()}usdt@kline_{financial_params['interval']}" for symbol in financial_params['symbols']],
        "id": 1
    }
    ws.send(json.dumps(params))
    logging.info(f"Subscribed to {', '.join(financial_params['symbols'])} with interval {financial_params['interval']}")

    # Initialize the environment with live data
    live_env = DummyVecEnv([lambda: CryptoTradingEnv(reshaped_data, mapping, sorted(valid_symbols), timestamps, financial_params)])
    live_env = VecNormalize(live_env, norm_obs=True, norm_reward=True)
    logging.debug("Environment initialized and normalized")

    # Load the trained model
    # model = PPO.load(model_path)
    # Load all models from the directory
    model_files = glob.glob(os.path.join(model_path, "model_ppo_crypto_trading*.zip"))
    models = []
    for model_file in model_files:
        # live_env = DummyVecEnv([lambda: CryptoTradingEnv(reshaped_data, mapping, sorted(valid_symbols), timestamps, financial_params)])
        model = PPO.load(model_file, env=live_env)
        model.policy.eval()
        models.append(model)
    logging.info(f"Models loaded from {model_path}") 
    


# def process_ticker(data):
#     symbol = data['symbol']
#     price = data['price']
#     # Update your environment with the new data
#     live_env.envs[0].update_data({symbol: price})
    
#     # Make a prediction
#     obs = live_env.reset()
#     action, _states = model.predict(obs)
    
#     # Get additional info from the environment
#     _, _, _, info = live_env.step(action)
    
#     # Execute the trade
#     execute_trade(symbol, action, info)
    
#     # Log the trade
#     logging.info(f"Executed trade for {symbol} with action {action} and info {info}")

# Main function to start the WebSocket connection
def start_trading_bot():
    websocket_url = os.getenv("WEBSOCKET_URL")
    logging.info(f"Connecting to WebSocket at {websocket_url}")
    ws = WebSocketApp(websocket_url, 
                      on_message=on_message, 
                      on_error=on_error, 
                      on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()
    
import logging
from logging.handlers import RotatingFileHandler

# Configure logging with INFO level to stdout and DEBUG level to a file
def setup_logging():
    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels of logs

    # Create a handler for DEBUG level logs to a file
    debug_handler = RotatingFileHandler('debug.log', maxBytes=1048576, backupCount=5)
    debug_handler.setLevel(logging.DEBUG)

    # Create a handler for INFO level logs to a different file
    info_handler = RotatingFileHandler('info.log', maxBytes=1048576, backupCount=5)
    info_handler.setLevel(logging.INFO)

    # Create formatters and add them to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    debug_handler.setFormatter(formatter)
    info_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(debug_handler)
    logger.addHandler(info_handler)

if __name__ == "__main__":
    global model_path
    parser = argparse.ArgumentParser(description='Live trading for a trained PPO model on crypto.')
    parser.add_argument('-m', '--model_path', type=str, help='Path to the trained model file')
    args = parser.parse_args()
    model_path = args.model_path

    # Configure logging to write to a file
    # logging.basicConfig(filename='live_trading.log', level=logging.INFO, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
    setup_logging()
    start_trading_bot()