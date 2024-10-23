import logging
from bayes_opt import BayesianOptimization
from tabulate import tabulate
import hashlib
import argparse  # Add this import
from sklearn.model_selection import KFold

# Set up logging
logging.basicConfig(level=logging.INFO)

from trading_bot import financial_params, ppo_hyperparams, prepare, train, test

# Global dictionary to store the output directory for each configuration
output_dirs = {}

def create_config_id(params):
    """Create a unique identifier for the given parameters."""
    params_str = str(sorted(params.items()))
    return hashlib.md5(params_str.encode()).hexdigest()

def train_and_evaluate(kelly_fraction, tp_atr_mult, sl_atr_mult, tp_percentage, sl_percentage, total_timesteps, risk_per_trade, interval, limit, initial_balance, cooldown_period, leverage_min, leverage_max, n_steps_index, batch_size_index, n_epochs, learning_rate, clip_range, gae_lambda, ent_coef, vf_coef, max_grad_norm, boost_leverage, normalize_reward, n_splits=2):
    global output_dirs  # Declare the global dictionary
    
    # Create a unique identifier for the current configuration
    params = {
        'kelly_fraction': kelly_fraction,
        'tp_atr_mult': tp_atr_mult,
        'sl_atr_mult': sl_atr_mult,
        'tp_percentage': tp_percentage,
        'sl_percentage': sl_percentage,
        'total_timesteps': total_timesteps,
        'risk_per_trade': risk_per_trade,
        'interval': interval,
        'limit': limit,
        'initial_balance': initial_balance,
        'cooldown_period': cooldown_period,
        'leverage_min': leverage_min,
        'leverage_max': leverage_max,
        'boost_leverage': boost_leverage,
        'normalize_reward': normalize_reward,
        'n_steps_index': n_steps_index,
        'batch_size_index': batch_size_index,
        'n_epochs': n_epochs,
        'learning_rate': learning_rate,
        'clip_range': clip_range,
        'gae_lambda': gae_lambda,
        'ent_coef': ent_coef,
        'vf_coef': vf_coef,
        'max_grad_norm': max_grad_norm
    }
    config_id = create_config_id(params)
    
    # Define the list of n_steps and batch sizes (powers of 2)
    n_steps_list = [512, 1024, 2048, 4096]
    batch_sizes = [32, 64, 128, 256]
    
    n_steps = n_steps_list[int(round(n_steps_index))]
    batch_size = batch_sizes[int(round(batch_size_index))]
    
    total_timesteps = int(total_timesteps)
    risk_per_trade = float(risk_per_trade)
    interval = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'][int(round(interval))]
    limit = int(limit)
    initial_balance = float(initial_balance)
    cooldown_period = int(cooldown_period)
    leverage_min = float(leverage_min)
    leverage_max = float(leverage_max)
    boost_leverage = bool(boost_leverage)
    normalize_reward = bool(normalize_reward)
    params = financial_params
    
    # Update params object
    params.update({
        'kelly_fraction': kelly_fraction,
        'tp_atr_mult': tp_atr_mult,
        'sl_atr_mult': sl_atr_mult,
        'tp_percentage': tp_percentage,
        'sl_percentage': sl_percentage,
        'total_timesteps': total_timesteps,
        'risk_per_trade': risk_per_trade,
        'interval': interval,
        'limit': limit,
        'initial_balance': initial_balance,
        'cooldown_period': cooldown_period,
        'leverage_min': leverage_min,
        'leverage_max': leverage_max,
        'boost_leverage': boost_leverage,
        'normalize_reward': normalize_reward
    })
    
    # Update PPO hyperparameters
    ppo_hyperparams.update({
        'n_steps': n_steps,  # Use the selected n_steps
        'batch_size': batch_size,  # Use the selected batch size
        'n_epochs': int(n_epochs),
        'learning_rate': float(learning_rate),
        'clip_range': float(clip_range),
        'gae_lambda': float(gae_lambda),
        'ent_coef': float(ent_coef),
        'vf_coef': float(vf_coef),
        'max_grad_norm': float(max_grad_norm)
    })
    
    data, mapping, valid_symbols, timestamps, _ = prepare(params)
    valid_symbols = sorted(valid_symbols)
    
    if data is not None:
        kf = KFold(n_splits=n_splits)
        scores = []

        for train_index, test_index in kf.split(data):
            train_data, test_data = data[train_index], data[test_index]
            train_timestamps, test_timestamps = timestamps[train_index], timestamps[test_index]

            # Train the model
            model, output_dir = train(train_data, mapping, valid_symbols, train_timestamps, params)
            
            if model:
                # retries = 0
                # while retries < 5:
                test_performance_score, num_trades = test(model, test_data, mapping, test_timestamps, valid_symbols, output_dir, params)
                    # if num_trades >= 25:
                scores.append(test_performance_score)
                        # break
                    # retries += 1

        if scores:
            # Store the output directory in the global dictionary
            output_dirs[config_id] = output_dir
            return sum(scores) / len(scores)  # Return the average performance score
    return 0  # Return 0 if no data or model

def optimize_hyperparameters(init_points, n_iter, process_length, n_splits):
    
    pbounds = {
        'kelly_fraction': (0.1, 0.5),
        'tp_atr_mult': (1, 4),
        'sl_atr_mult': (1, 2),
        'tp_percentage': (0.01, 0.2),
        'sl_percentage': (0.01, 0.1),
        'total_timesteps': (10_000, 50_000),
        'risk_per_trade': (0.01, 0.2),
        'interval': (0, 5),
        'limit': (500, 1000),
        'initial_balance': (5_000, 10_000),
        'cooldown_period': (1, 20),
        'leverage_min': (1, 20),
        'leverage_max': (20, 150),
        'n_steps_index': (0, 3),  # Index for n_steps
        'batch_size_index': (0, 3),  # Index for batch sizes
        'n_epochs': (3, 30),
        'learning_rate': (1e-5, 1e-2),
        'clip_range': (0.1, 0.4),
        'gae_lambda': (0.8, 1.0),
        'ent_coef': (0.0, 0.1),
        'vf_coef': (0.1, 1.0),
        'max_grad_norm': (0.3, 1.0),
        'boost_leverage': (0, 1),  # Use 0 and 1 for categorical
        'normalize_reward': (0, 1)  # Use 0 and 1 for categorical
    }
    
    if process_length == 'long':
        pbounds = {
            'kelly_fraction': (0.2, 0.8),
            'tp_atr_mult': (1, 8),
            'sl_atr_mult': (1, 4),
            'tp_percentage': (0.01, 0.4),
            'sl_percentage': (0.01, 0.2),
            'total_timesteps': (20_000, 100_000),
            'risk_per_trade': (0.01, 0.2),
            'interval': (0, 5),
            'limit': (500, 1000),
            'initial_balance': (1_000, 10_000),
            'cooldown_period': (1, 50),
            'leverage_min': (1, 20),
            'leverage_max': (20, 150),
            'n_steps_index': (0, 3),  # Index for n_steps
            'batch_size_index': (0, 3),  # Index for batch sizes
            'n_epochs': (3, 30),
            'learning_rate': (1e-5, 1e-2),
            'clip_range': (0.1, 0.4),
            'gae_lambda': (0.8, 1.0),
            'ent_coef': (0.0, 0.1),
            'vf_coef': (0.1, 1.0),
            'max_grad_norm': (0.3, 1.0),
            'boost_leverage': (0, 1),  # Use 0 and 1 for categorical
            'normalize_reward': (0, 1)  # Use 0 and 1 for categorical
        }
        
    if process_length == 'short':
        n_splits = 3
    elif process_length == 'long':
        n_splits = 5
    else:
        n_splits = 2

    optimizer = BayesianOptimization(
        f=lambda **params: train_and_evaluate(**params, n_splits=n_splits),
        pbounds=pbounds,
        random_state=1,
        verbose=2,
        allow_duplicate_points=True,  # Allow duplicate points for categorical
    )

    if process_length == 'short':
        init_points = 5
        n_iter = 25
    elif process_length == 'long':
        init_points = 10
        n_iter = 50
    else:
        init_points = 2
        n_iter = 3

    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter,
    )

    # Sort and list the top configurations
    results = sorted(optimizer.res, key=lambda x: x['target'], reverse=True)
    top_n = 10  # Number of top configurations to list
    for i, result in enumerate(results[:top_n]):
        config_id = create_config_id(result['params'])
        output_dir = output_dirs.get(config_id, "N/A")
        result_with_output_dir = {**result, 'output_dir': output_dir}
        print(f"params_{i+1} = {result_with_output_dir}")

    logging.info(f"\nBest parameters found: {optimizer.max}")

import os
def main():
    parser = argparse.ArgumentParser(description="Optimize hyperparameters for the trading bot.")
    parser.add_argument('-i', '--init_points', type=int, help='Number of initial points for Bayesian Optimization')
    parser.add_argument('-n', '--n_iter', type=int, help='Number of iterations for Bayesian Optimization')
    parser.add_argument('-p', '--process_length', choices=['short', 'long'], help='Length of the optimization process')
    parser.add_argument('-s', '--n_splits', type=int, help='Number of splits for cross-validation')

    args = parser.parse_args()

    os.system('say "Dear Mr Sursock, the script has started optimizing your bot"')
    logging.info("Script started")
    
    optimize_hyperparameters(args.init_points, args.n_iter, args.process_length, args.n_splits)
    
    logging.info("Script finished")
    os.system('say "Dear Mr Sursock, the script has finished optimizing your bot"')

if __name__ == "__main__":
    main()
