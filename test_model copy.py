import logging
import os
import sys
import argparse
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from environment import CryptoTradingEnv
from reporting import plot_sl_tp_distribution, plot_cumulative_returns, plot_histograms, plot_metrics, get_trade_metrics, display_trade_metrics, plot_stock_data
from parameters import create_financial_params, selected_params
from trading_bot import prepare  # Assuming prepare is in trading_bot.py
import numpy as np
from feature_analysis import calculate_feature_importance, create_summary_feature_importance_plot, create_symbol_feature_importance_plots
import random
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)

class DualOutput:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def perform_feature_analysis(test_data, mapping, valid_symbols, model, params, output_dir, method='shap'):
    logging.info("Starting feature importance analysis")
    
    # Create a 'features' folder within the output_dir
    features_dir = os.path.join(output_dir, 'features')
    os.makedirs(features_dir, exist_ok=True)
    
    # Prepare features (X) and target variable (y)
    num_candles, num_symbols, num_features = test_data.shape
    X = test_data[:-1].reshape(num_candles - 1, num_symbols * num_features)
    y = calculate_future_returns(test_data, mapping)

    # Create meaningful feature names
    feature_names = []
    for symbol in valid_symbols:
        for feature, index in mapping.items():
            feature_names.append(f"{symbol}_{feature}")

    # Calculate feature importance
    feature_importance = calculate_feature_importance(X, y, model, feature_names=feature_names, method=method)

    # Create and save feature importance plots in the features directory
    create_symbol_feature_importance_plots(feature_importance, feature_names, valid_symbols, method, features_dir)
    create_summary_feature_importance_plot(feature_importance, feature_names, valid_symbols, method, features_dir)

    # Print top 10 most important features
    top_features = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)[:10]
    print(f"Top 10 most important features ({method} method):")
    for name, importance in top_features:
        print(f"{name}: {importance:.2%}")

    logging.info("Feature importance analysis completed successfully.")

def calculate_future_returns(data, mapping):
    future_returns = np.zeros(len(data) - 1)
    for i in range(len(data) - 1):
        future_returns[i] = np.mean((data[i+1, :, mapping['close']] - data[i, :, mapping['close']]) / data[i, :, mapping['close']])
    return future_returns

def test_model(test_data, mapping, timestamps, valid_symbols, model, params, output_dir):
    logging.info('Started testing...')
    
    test_env = DummyVecEnv([lambda: CryptoTradingEnv(test_data, mapping, valid_symbols, timestamps, params)])

    obs = test_env.reset()
    done = False
    step_count = 0
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = test_env.step(action)
        step_count += 1
        if step_count % 10 == 0 or done:
            logging.debug(f"Step {step_count}: reward = {rewards[0]:.2f}, done = {done}")
            if 'info' in info[0]:
                logging.info(f"Info: {info[0]['info']}")

        if done:
            logging.info(f"Episode terminated at step {step_count}. Reason: {info[0].get('terminal_info', 'Unknown')}")

    logging.info(f"Testing completed after {step_count} steps")

    # Check trade history
    trade_history = test_env.envs[0].trade_history
    logging.info(f"Total trades recorded: {len(trade_history)}")
    if len(trade_history) > 0:
        logging.debug(f"First trade: {trade_history[0]}")
        logging.debug(f"Last trade: {trade_history[-1]}")
    else:
        logging.warning("No trades were recorded during testing!")

    # Display trade metrics
    testing_output_dir = output_dir
    
    # Plot cumulative returns
    plot_cumulative_returns(test_env.envs[0].trade_history, test_data, mapping, valid_symbols, 
                            save_path=testing_output_dir + '/cumulative_returns.png')

    plot_histograms(trade_history, test_env.envs[0].rewards, test_env.envs[0].actions, testing_output_dir)
    plot_metrics(trade_history, testing_output_dir)
    plot_stock_data(test_data, valid_symbols, mapping, timestamps, trade_history, 'Testing Events', testing_output_dir)
    print('Testing Report')
    display_trade_metrics(test_env.envs[0])
    plot_sl_tp_distribution(test_env.envs[0].get_sl_tp_distribution(), save_path=testing_output_dir + '/sl_tp_distribution.png')
    # Calculate performance score using net profit
    trade_metrics = get_trade_metrics(trade_history, test_env.envs[0].symbols)
    total_trades = len(trade_history)

    if total_trades < 50:
        logging.warning("Not enough trades to evaluate performance. Skipping this test.")
        return None, None, trade_history

    net_profit = trade_metrics.get('PF', {}).get('net_profit', 0)
    performance_score = net_profit / params['initial_balance'] * 100
    performance_score = round(performance_score, 2)
    
    print(f"Test Performance Score (Strategy Return): {performance_score}%")

    return net_profit, performance_score, trade_history

def run_multiple_tests(test_data, mapping, timestamps, valid_symbols, model_path, params, num_tests, feature_method='shap'):
    net_profits = []
    performance_scores = []
    trade_counts = []

    # Create a timestamped directory for the test run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.dirname(model_path)
    test_run_dir = os.path.join(output_dir, f'testing_{timestamp}')
    os.makedirs(test_run_dir, exist_ok=True)
    
    # Redirect stdout to both console and logfile in test_run_dir
    logfile_path = os.path.join(test_run_dir, "test_log.txt")
    sys.stdout = DualOutput(logfile_path)

    # Set seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Load the model once
    test_env = DummyVecEnv([lambda: CryptoTradingEnv(test_data, mapping, valid_symbols, timestamps, params)])
    model = PPO.load(model_path, env=test_env)

    # Ensure the model is in evaluation mode
    model.policy.eval()

    # Perform feature analysis only once
    if feature_method is not None:
        perform_feature_analysis(test_data, mapping, valid_symbols, model, params, test_run_dir, feature_method)

    for test_number in range(1, num_tests + 1):
        logging.info(f"Starting test {test_number}")
        
        test_number_dir = os.path.join(test_run_dir, f'test_{test_number}')
        os.makedirs(test_number_dir, exist_ok=True)

        # Reset the environment
        test_env.reset()

        net_profit, performance_score, trade_history = test_model(test_data, mapping, timestamps, valid_symbols, model, params, test_number_dir)
        
        # Check trade history immediately after test_model
        trade_count = len(trade_history)
        logging.info(f"Trade history length: {trade_count}")
        if trade_count > 0:
            logging.debug(f"First trade: {trade_history[0]}")
            logging.debug(f"Last trade: {trade_history[-1]}")
        else:
            logging.warning("No trades were recorded in this test!")

        if net_profit is not None and performance_score is not None:
            net_profits.append(net_profit)
            performance_scores.append(performance_score)
            trade_counts.append(trade_count)
            logging.info(f"Test {test_number} completed. Trades: {trade_count}, Net Profit: {net_profit}, Performance Score: {performance_score}")
        else:
            logging.warning(f"Test {test_number} skipped due to insufficient trades. Trades: {trade_count}")
    
    num_valid_tests = len(net_profits)

    # Log summary statistics
    logging.info(f"Summary of {num_valid_tests} tests:")
    logging.info(f"Average number of trades: {np.mean(trade_counts):.2f} (std: {np.std(trade_counts):.2f})")
    logging.info(f"Average net profit: {np.mean(net_profits):.2f} (std: {np.std(net_profits):.2f})")
    logging.info(f"Average performance score: {np.mean(performance_scores):.2f}% (std: {np.std(performance_scores):.2f}%)")

    # Plotting
    import matplotlib.pyplot as plt

    x = range(1, num_valid_tests + 1)   
    fig, ax1 = plt.subplots(figsize=(15, 10))

    # Plot net profit as a line
    ax1.set_xlabel('Test Number')
    ax1.set_ylabel('Net Profit', color='tab:blue')
    ax1.plot(x, net_profits, color='tab:blue', label='Net Profit', marker='o')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Create a second y-axis for performance score
    ax2 = ax1.twinx()
    ax2.set_ylabel('Strategy Return (%)', color='tab:orange')
    ax2.bar(x, performance_scores, color='tab:orange', alpha=0.6, label='Strategy Return')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Save the figure to the test run directory
    plt.title('Net Profit and Strategy Return for Multiple Tests')
    plt.savefig(os.path.join(test_run_dir, "test_results.png"))  # Save the figure
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a trained PPO model on crypto trading environment.')
    parser.add_argument('-m', '--model_path', type=str, help='Path to the trained model file')
    parser.add_argument('-n', '--num_tests', type=int, default='1', help='Number of tests for evaluation')
    parser.add_argument('-f', '--feature_method', choices=['shap', 'perm', 'grad', 'tree'], default='shap',
                        help="Method to calculate feature importance")
    args = parser.parse_args()

    # Example parameters (replace with actual values)
    financial_params = create_financial_params(selected_params)
    # financial_params['initial_balance'] = 5000
    financial_params['symbols'] = ['GAINS_20']
    # financial_params['interval'] = '4h'
    # financial_params['limit'] = 250 
    # financial_params['leverage_max'] = 1
    # financial_params['leverage_min'] = 1
    # financial_params['risk_per_trade'] = 0.01
    # financial_params['symbols'] = ['EGLD', 'METIS', 'SAND', 'CHZ', 'LUNA', 'RDNT', 'ZRO', 'BANANA', 'CELO', 'FXS'] #['GAINS_20']
    
    # financial_params = {
    #     'symbols': ['GAINS_20'], #['BTC', 'ETH', 'BNB', 'SOL', 'NEAR', 'FTM', 'ADA'], #, 'LINK', 'SHIB', 'BONK'],
    #     'min_collateral': 50,  # TODO check if it works
    #     'cooldown_period': 16, 
    #     'initial_balance': 10_000, 
    #     'interval': '15m', 
    #     'leverage_max': 150, 
    #     'leverage_min': 1, 
    #     'limit': 1000, 
    #     'risk_per_trade': 0.01, 
    #     'total_timesteps': 20000,
    # }
    
    # Prepare data
    data, mapping, valid_symbols, timestamps, _ = prepare(financial_params)
    valid_symbols = sorted(valid_symbols)
    
    if data is not None:
        if data.shape[0] != financial_params['limit']:
            logging.info(f"Data shape does not match limit. Expected {financial_params['limit']}, got {data.shape[0]}")
            sys.exit(1)
        
        # Split data into training and testing sets
        # split_index = int(len(data) * 0.5)
        test_data = data #[split_index:]
        test_timestamps = timestamps #[split_index:]

        model_path = args.model_path

        # Redirect stdout to both console and logfile
        logfile_path = os.path.join(os.path.dirname(model_path), "test_log.txt")
        sys.stdout = DualOutput(logfile_path)
        
        num_tests = args.num_tests  # Specify the number of tests to run
        feature_method = None
        if args.feature_method:
            feature_method = args.feature_method
        run_multiple_tests(test_data, mapping, test_timestamps, valid_symbols, model_path, financial_params, num_tests, feature_method)

        # Restore the original stdout
        sys.stdout.log.close()
        sys.stdout = sys.stdout.terminal

        # logging.info(f"Test Performance Score: {performance_score}%")
    else:
        logging.error("Data preparation failed. Exiting.")