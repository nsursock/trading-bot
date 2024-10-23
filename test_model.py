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
import glob
from sklearn.linear_model import LogisticRegression

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

def run_multiple_tests(test_data, mapping, timestamps, valid_symbols, model_dir, params, num_tests, feature_method='shap', action_selection_method='voting'):
    net_profits = []
    performance_scores = []
    trade_counts = []

    # Create a timestamped directory for the test run, including the action selection method
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = model_dir
    test_run_dir = os.path.join(output_dir, f'testing_{timestamp}_{action_selection_method}')
    os.makedirs(test_run_dir, exist_ok=True)
    
    # Redirect stdout to both console and logfile in test_run_dir
    logfile_path = os.path.join(test_run_dir, "test_log.txt")
    sys.stdout = DualOutput(logfile_path)

    # # Set seeds for reproducibility
    # seed = 42
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)

    # Load all models from the directory
    model_files = glob.glob(os.path.join(model_dir, "model_ppo_crypto_trading*.zip"))
    models = []
    for model_file in model_files:
        test_env = DummyVecEnv([lambda: CryptoTradingEnv(test_data, mapping, valid_symbols, timestamps, params)])
        model = PPO.load(model_file, env=test_env)
        model.policy.eval()
        models.append(model)

    # Initialize a simple meta-model for stacking
    if action_selection_method == 'stacking':
        meta_model = LogisticRegression()

    # Perform feature analysis only once
    if feature_method != 'none':
        perform_feature_analysis(test_data, mapping, valid_symbols, models[0], params, test_run_dir, feature_method)

    for test_number in range(1, num_tests + 1):
        print('---------------------------------------------------------------------------')
        print(f"Starting test {test_number}")
        
        test_number_dir = os.path.join(test_run_dir, f'test_{test_number}')
        os.makedirs(test_number_dir, exist_ok=True)

        # Reset the environment and related variables
        test_env = DummyVecEnv([lambda: CryptoTradingEnv(test_data, mapping, valid_symbols, timestamps, params)])
        obs = test_env.reset()
        done = False
        step_count = 0

        # Use ensemble of models to predict actions
        while not done:
            # Aggregate predictions from all models
            actions = [model.predict(obs)[0] for model in models]
            
            if action_selection_method == 'bagging':
                # Implement bagging by randomly sampling actions with replacement
                sampled_actions = random.choices(actions, k=len(actions))
                # Convert actions to tuples to make them hashable
                actions_as_tuples = [tuple(map(tuple, action)) if isinstance(action, (list, np.ndarray)) else tuple(action) for action in sampled_actions]
                action = max(set(actions_as_tuples), key=actions_as_tuples.count)
                action = np.array(action)
            
            elif action_selection_method == 'stacking':
                # Convert actions to a format suitable for the meta-model
                actions_array = np.array(actions).reshape(len(actions), -1)
                
                # Train the meta-model on the aggregated actions
                # Modify this to work with models instead of symbols
                # Here, we assume that the target is the action with the highest confidence across all models
                # This is a placeholder logic; adjust it based on your specific requirements
                targets = np.array([np.argmax(action) for action in actions_array.T])  # Transpose to focus on models

                meta_model.fit(actions_array.T, targets)
                
                # Predict the final action for each symbol using the meta-model
                predicted_actions = []
                for symbol_idx in range(len(valid_symbols)):
                    # Extract actions for the current symbol across all models
                    symbol_actions = actions_array[:, symbol_idx]
                    # Predict the action for the current symbol
                    predicted_action = meta_model.predict(symbol_actions.reshape(1, -1))
                    predicted_actions.append(predicted_action[0])  # Ensure it's a scalar

                # Ensure the predicted actions form an array of length num_symbols
                action = np.array(predicted_actions)  # Convert to array

                # Clip the predicted actions to be within the range [0, 3]
                action = np.clip(action, 0, 3)

                action = [action]
            
            elif action_selection_method == 'voting':
                # Convert actions to tuples to make them hashable
                actions_as_tuples = [tuple(map(tuple, action)) if isinstance(action, (list, np.ndarray)) else tuple(action) for action in actions]
                action = max(set(actions_as_tuples), key=actions_as_tuples.count)
                action = np.array(action)
            
            elif action_selection_method == 'averaging':
                action = np.mean(actions, axis=0)
            
            elif action_selection_method == 'mediating':
                action = np.median(actions, axis=0)
            
            elif action_selection_method == 'maximizing':
                action = np.max(actions, axis=0)
            
            elif action_selection_method == 'minimizing':
                action = np.min(actions, axis=0)
            
            else:
                raise ValueError(f"Unknown action selection method: {action_selection_method}")
            
            obs, rewards, done, info = test_env.step(action)
            step_count += 1
            if step_count % 10 == 0 or done:
                logging.debug(f"Step {step_count}: reward = {rewards[0]:.2f}, done = {done}")
                if 'info' in info[0]:
                    logging.info(f"Info: {info[0]['info']}")

            if done:
                logging.info(f"Episode terminated at step {step_count}. Reason: {info[0].get('terminal_info', 'Unknown')}")

        # Check trade history
        trade_history = test_env.envs[0].trade_history
        logging.info(f"Total trades recorded: {len(trade_history)}")
        if len(trade_history) > 0:
            logging.debug(f"First trade: {trade_history[0]}")
            logging.debug(f"Last trade: {trade_history[-1]}")
        else:
            logging.warning("No trades were recorded during testing!")

        # Display trade metrics
        testing_output_dir = test_number_dir
        
        # Plot cumulative returns
        plot_cumulative_returns(test_env.envs[0].trade_history, test_data, mapping, valid_symbols, 
                                save_path=testing_output_dir + '/cumulative_returns.png')

        plot_histograms(trade_history, test_env.envs[0].rewards, test_env.envs[0].actions, testing_output_dir)
        plot_metrics(trade_history, testing_output_dir)
        # plot_stock_data(test_data, valid_symbols, mapping, timestamps, trade_history, 'Testing Events', testing_output_dir)
        print('Testing Report')
        display_trade_metrics(test_env.envs[0])
        plot_sl_tp_distribution(test_env.envs[0].get_sl_tp_distribution(), save_path=testing_output_dir + '/sl_tp_distribution.png')
        # Calculate performance score using net profit
        trade_metrics = get_trade_metrics(trade_history, test_env.envs[0].symbols)
        total_trades = len(trade_history)

        if total_trades < 50:
            logging.warning("Not enough trades to evaluate performance. Skipping this test.")
            continue

        net_profit = trade_metrics.get('PF', {}).get('net_profit', 0)
        performance_score = net_profit / params['initial_balance'] * 100
        performance_score = round(performance_score, 2)
        
        print(f"Test Performance Score (Strategy Return): {performance_score}%")

        net_profits.append(net_profit)
        performance_scores.append(performance_score)
        trade_counts.append(total_trades)
        logging.info(f"Test {test_number} completed. Trades: {total_trades}, Net Profit: {net_profit}, Performance Score: {performance_score}")

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
    parser.add_argument('-m', '--model_dir', type=str, help='Path to the directory containing trained models')
    parser.add_argument('-n', '--num_tests', type=int, default=1, help='Number of tests for evaluation')
    parser.add_argument('-f', '--feature_method', choices=['none', 'shap', 'perm', 'grad', 'tree'], default='none',
                        help="Method to calculate feature importance")
    parser.add_argument('-a', '--action_selection_method', choices=['bagging', 'stacking', 'voting', 'averaging', 'mediating', 'maximizing', 'minimizing', 'all'], default='all',
                        help="Method to aggregate actions from multiple models")
    args = parser.parse_args()

    # Example parameters (replace with actual values)
    financial_params = create_financial_params(selected_params)
    financial_params['initial_balance'] = 5000
    # financial_params['boost_leverage'] = False
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

        model_dir = args.model_dir

        # Redirect stdout to both console and logfile
        logfile_path = os.path.join(os.path.dirname(model_dir), "test_log.txt")
        sys.stdout = DualOutput(logfile_path)
        
        num_tests = args.num_tests  # Specify the number of tests to run
        feature_method = None
        if args.feature_method:
            feature_method = args.feature_method
            
        if args.action_selection_method == 'all':
            action_methods = ['bagging', 'stacking', 'voting', 'averaging', 'mediating', 'maximizing', 'minimizing']
            for method in action_methods:
                logging.info(f"Running tests with action selection method: {method}")
                run_multiple_tests(test_data, mapping, test_timestamps, valid_symbols, model_dir, financial_params, num_tests, feature_method, method)
        else:
            run_multiple_tests(test_data, mapping, test_timestamps, valid_symbols, model_dir, financial_params, num_tests, feature_method, args.action_selection_method)

        # Restore the original stdout
        sys.stdout.log.close()
        sys.stdout = sys.stdout.terminal

        # logging.info(f"Test Performance Score: {performance_score}%")
    else:
        logging.error("Data preparation failed. Exiting.")