from stable_baselines3 import PPO
# import numpy as np
# import pandas as pd
import logging
import random
import requests
# import matplotlib.pyplot as plt
import os
import time
from tabulate import tabulate
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from early_stopping_callback import EarlyStoppingCallback
# import profile
import sys
from parameters import create_financial_params, create_ppo_hyperparams, selected_params
from sklearn.model_selection import KFold

# Set up logging
logging.basicConfig(level=logging.INFO)

# # Example usage
financial_params = create_financial_params(selected_params)
ppo_hyperparams = create_ppo_hyperparams(selected_params)

def log_parameters(params):
    table = [[key, value] for key, value in params.items() if key != 'symbols']
    print("Parameters:\n" + tabulate(table, headers=["Parameter", "Value"], tablefmt="pretty"))

def fetch_symbols():
    response = requests.get('https://backend-arbitrum.gains.trade/trading-variables')
    pairs = response.json()['pairs']
    return [{'symbol': pair['from'], 'index': idx, 'groupIndex': pair['groupIndex']} for idx, pair in enumerate(pairs)]

def select_cryptos(count):
    symbols = fetch_symbols()
    filtered_symbols = [symbol['symbol'] for symbol in symbols if 'groupIndex' in symbol and int(symbol['groupIndex']) in [0, 10]]
    random.shuffle(filtered_symbols)
    selected_cryptos = filtered_symbols[:count]
    return selected_cryptos

from collections import Counter
import random

def prepare(params):
    logging.info('Started fetching...')
    from utilities import fetch_binance_klines, preprocess_data  # Import the required functions

    symbols = sorted(params['symbols'])
    interval = params['interval']
    limit = params['limit']

    if symbols[0].startswith('GAINS_'):
        count = int(symbols[0].split('_')[1])
        symbols = select_cryptos(count)
        logging.info(f"Selected top {count} cryptos: {symbols}")

    valid_symbols = []
    dfs = []
    time_ranges = []

    # Dictionary to convert intervals to days
    interval_to_days = {
        '1m': 1 / (24 * 60),
        '3m': 3 / (24 * 60),
        '5m': 5 / (24 * 60),
        '15m': 15 / (24 * 60),
        '30m': 30 / (24 * 60),
        '1h': 1 / 24,
        '2h': 2 / 24,
        '4h': 4 / 24,
        '6h': 6 / 24,
        '8h': 8 / 24,
        '12h': 12 / 24,
        '1d': 1,
        '3d': 3,
        '1w': 7,
        '1M': 30  # Assuming 1 month is approximately 30 days
    }

    # Calculate and display the number of days covered by the interval and limit
    if interval in interval_to_days:
        days_covered = limit * interval_to_days[interval]
        logging.info(f"Data covers approximately {days_covered:.2f} days with interval '{interval}' and limit {limit}")
    else:
        logging.error(f"Interval '{interval}' is not recognized.")

    for symbol in symbols:
        try:
            df = fetch_binance_klines(symbol + 'USDT', interval, limit)
            if not df.empty:
                time_range = (df.index.min(), df.index.max())
                time_ranges.append(time_range)
                valid_symbols.append(symbol)
                dfs.append(df)
                logging.debug(f"Number of valid symbols {len(valid_symbols)}")
        except Exception as e:
            logging.warning(f"Data not available for {symbol}. Skipping... (fyi, error: {e})")

    if not valid_symbols:
        logging.error("No valid symbols found. Exiting preparation.")
        return None, None, None, None, None

    # Find the most common time range
    most_common_range = Counter(time_ranges).most_common(1)[0][0]

    # Filter symbols and dataframes based on the most common time range
    filtered_symbols = []
    filtered_dfs = []
    for symbol, df, time_range in zip(valid_symbols, dfs, time_ranges):
        if time_range == most_common_range:
            filtered_symbols.append(symbol)
            filtered_dfs.append(df)

    # Select 10 random symbols from the filtered list
    num_symbols = 10
    if len(filtered_symbols) > num_symbols:
        selected_indices = random.sample(range(len(filtered_symbols)), num_symbols)
        selected_symbols = [filtered_symbols[i] for i in selected_indices]
        selected_dfs = [filtered_dfs[i] for i in selected_indices]
    else:
        selected_symbols = filtered_symbols
        selected_dfs = filtered_dfs

    logging.info(f"Selected symbols: {selected_symbols}")
    data, mapping, timestamps = preprocess_data(selected_dfs, selected_symbols)
    return data, mapping, selected_symbols, timestamps, selected_dfs

class DualOutput:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

original_stdout = sys.stdout  # Save the original stdout

from early_stopping_callback import EarlyStoppingCallback

def train(train_data, mapping, valid_symbols, timestamps, params):
    logging.info('Started training...')
    
    # Log parameters before training
    log_parameters(params)
    log_parameters(ppo_hyperparams)
    
    model_path = './logs'
    
    # Create a directory with the current timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(model_path, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Redirect stdout to both console and logfile
    logfile_path = os.path.join(output_dir, "trading_bot_log.txt")
    sys.stdout = DualOutput(logfile_path)
        
    from reporting import display_trade_metrics, plot_histograms, plot_metrics
    from environment import CryptoTradingEnv

    train_env = DummyVecEnv([lambda: CryptoTradingEnv(train_data, mapping, valid_symbols, timestamps, params)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
    
    # Initialize the PPO agent with custom hyperparameters
    model = PPO('MlpPolicy', train_env, verbose=1, **ppo_hyperparams)

    # Set up the evaluation environment
    eval_env = DummyVecEnv([lambda: CryptoTradingEnv(train_data, mapping, valid_symbols, timestamps, params)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)

    # Define the EarlyStoppingCallback
    early_stopping_callback = EarlyStoppingCallback(eval_env, eval_freq=10000, patience=5, verbose=1)

    # Train the agent with the callback
    model.learn(total_timesteps=params['total_timesteps'], callback=early_stopping_callback)

    # Save the model
    model.save(os.path.join(output_dir, "model_ppo_crypto_trading"))

    # Log parameters before training
    log_parameters(params)
    log_parameters(ppo_hyperparams)
    print('Training Report')
    
    # Display trade metrics
    training_output_dir = os.path.join(output_dir, 'training')
    os.makedirs(training_output_dir, exist_ok=True)
    plot_histograms(train_env.envs[0].trade_history, train_env.envs[0].rewards, train_env.envs[0].actions, training_output_dir)
    plot_metrics(train_env.envs[0].trade_history, training_output_dir)
    display_trade_metrics(train_env.envs[0])
    
    # Restore the original stdout
    sys.stdout.log.close()
    sys.stdout = original_stdout

    return model, output_dir

def test(model, test_data, mapping, timestamps, valid_symbols, output_dir, params, repetition=None):
    # Redirect stdout to both console and logfile
    logfile_path = os.path.join(output_dir, "trading_bot_log.txt")
    sys.stdout = DualOutput(logfile_path)
    
    logging.info('Started testing...')
    
    from environment import CryptoTradingEnv
    from reporting import plot_histograms, plot_metrics, get_trade_metrics, display_trade_metrics, plot_stock_data

    test_env = DummyVecEnv([lambda: CryptoTradingEnv(test_data, mapping, valid_symbols, timestamps, params)])

    # Load the trained model
    model = PPO.load(output_dir + "/model_ppo_crypto_trading", env=test_env)

    # Run the model on the test environment
    obs = test_env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = test_env.step(action)

    # Check the shape of test_data
    logging.debug(f"Shape of test_data: {np.array(test_data).shape}")

    # Ensure test_data is a 3D numpy array
    if not isinstance(test_data, np.ndarray) or test_data.ndim != 3:
        raise ValueError("test_data must be a 3D numpy array.")

    # Display trade metrics
    unique_test_folder = '_' + str(repetition) if repetition != None else ''
    testing_output_dir = os.path.join(output_dir, 'testing' + unique_test_folder)
    os.makedirs(testing_output_dir, exist_ok=True)
    plot_histograms(test_env.envs[0].trade_history, test_env.envs[0].rewards, test_env.envs[0].actions, testing_output_dir)
    plot_metrics(test_env.envs[0].trade_history, testing_output_dir)
    plot_stock_data(test_data, valid_symbols, mapping, timestamps, test_env.envs[0].trade_history, 'Testing Events', testing_output_dir)
    print('Testing Report')
    display_trade_metrics(test_env.envs[0])

    # Save trade history to CSV
    # save_trade_history(test_env.envs[0], filename="test_trade_history.csv")

    # Calculate performance score using net profit
    trade_metrics = get_trade_metrics(test_env.envs[0].trade_history, test_env.envs[0].symbols)
    net_profit = trade_metrics.get('PF', {}).get('net_profit', 0)
    performance_score = net_profit / params['initial_balance'] * 100
    performance_score = round(performance_score, 2)

    print(f"Test Performance Score (Strategy Return): {performance_score}%")
    
    # Restore the original stdout
    sys.stdout.log.close()
    sys.stdout = original_stdout

    return performance_score, len(test_env.envs[0].trade_history)

import os
import numpy as np

def main():
    logging.info("Script started")
    os.system('say "Dear Mr Sursock, the script has started"')
    
    # financial_params['total_timesteps'] = 10000
    
    data, mapping, valid_symbols, timestamps, _ = prepare(financial_params)
    valid_symbols = sorted(valid_symbols)
    
    if data is not None:
        data = np.array(data)
        logging.info(f"Shape of data after conversion: {data.shape}")
        if data.ndim != 3:
            raise ValueError("Data must be a 3D numpy array after conversion.")

        kf = KFold(n_splits=5)
        fold = 1
        performance_scores = []
        models = []

        for train_index, test_index in kf.split(data):
            logging.info(f"Processing fold {fold}")
            train_data = data[train_index]
            test_data = data[test_index]
            train_timestamps = [timestamps[i] for i in train_index]
            test_timestamps = [timestamps[i] for i in test_index]

            model, output_dir = train(train_data, mapping, valid_symbols, train_timestamps, financial_params)
            if model:
                test_performance_score, _ = test(model, test_data, mapping, test_timestamps, valid_symbols, output_dir, financial_params)
                performance_scores.append(test_performance_score)
                models.append((model, test_performance_score, output_dir))

            fold += 1

        # Calculate average performance score
        average_performance_score = sum(performance_scores) / len(performance_scores)
        logging.info(f"Average Performance Score: {average_performance_score:.2f}%")

        # Select the best model based on performance score
        best_model, best_score, best_output_dir = max(models, key=lambda x: x[1])
        logging.info(f"Best Performance Score: {best_score:.2f}% from model in {best_output_dir}")

    os.system('say "Dear Mr Sursock, the script has finished"')
    logging.info("Script finished")

if __name__ == "__main__":
    main()
