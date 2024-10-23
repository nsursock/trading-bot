import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import os
from tabulate import tabulate
from utilities import net_profit, sharpe_ratio, max_drawdown, risk_to_reward_ratio
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import warnings
import urllib3
import logging

# Define a function to suppress the specific warning
def suppress_urllib3_warning():
    warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)

# Call this function at the very beginning of your script
suppress_urllib3_warning()

def plot_histograms(trade_history, rewards, actions_per_symbol, save_path='histograms.png'):
    collaterals = [trade['collateral'] for trade in trade_history]
    leverages = [trade['leverage'] for trade in trade_history]
    actions = np.concatenate(actions_per_symbol).tolist()

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    sns.histplot(collaterals, bins=20, kde=True, ax=axs[0, 0], color='blue')
    axs[0, 0].set_title('Collaterals')
    axs[0, 0].set_xlabel('Collateral')
    axs[0, 0].set_ylabel('Frequency')
    axs[0, 0].set_yscale('log')  # Set y-axis to log scale

    sns.histplot(leverages, bins=15, kde=True, ax=axs[0, 1], color='green')
    axs[0, 1].set_title('Leverages')
    axs[0, 1].set_xlabel('Leverage')
    axs[0, 1].set_ylabel('Frequency')
    axs[0, 1].set_yscale('log')  # Set y-axis to log scale

    sns.histplot(rewards, bins=20, kde=True, ax=axs[1, 0], color='red')
    axs[1, 0].set_title('Rewards')
    axs[1, 0].set_xlabel('Reward')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].set_yscale('log')  # Set y-axis to log scale

    sns.histplot(actions, bins=4, kde=False, ax=axs[1, 1], color='purple')
    axs[1, 1].set_title('Actions')
    axs[1, 1].set_xlabel('Action Type')
    axs[1, 1].set_ylabel('Frequency')
    axs[1, 1].set_yscale('log')  # Set y-axis to log scale

    plt.tight_layout()
    plt.savefig(save_path + '/histograms.png')  # Save the figure to a file
    plt.close()  # Close the figure to free up memory

    
def plot_metrics(trade_history, save_path='metrics.png'):
    symbol = 'Evolution of'
    returns = [trade['pnl'] / trade['collateral'] for trade in trade_history]
    cumulative_profits = np.cumsum([trade['pnl'] for trade in trade_history])
    sharpe_ratios = []
    max_drawdowns = []
    risk_to_reward_ratios = []

    for i in range(1, len(cumulative_profits)):
        sharpe_ratios.append(sharpe_ratio(returns[:i+1]))
        max_drawdowns.append(max_drawdown(returns[:i+1]))
        risk_to_reward_ratios.append(risk_to_reward_ratio(returns[:i+1]))

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)

    # Cumulative Returns
    axs[0, 0].plot(cumulative_profits, label='Cumulative Returns')
    axs[0, 0].set_title(f'{symbol} Cumulative Returns')
    axs[0, 0].set_ylabel('Cumulative Returns')
    axs[0, 0].legend()

    # Sharpe Ratio
    axs[0, 1].plot(sharpe_ratios, label='Sharpe Ratio', color='orange')
    axs[0, 1].set_title(f'{symbol} Sharpe Ratio')
    axs[0, 1].set_ylabel('Sharpe Ratio')
    axs[0, 1].legend()

    # Max Drawdown
    axs[1, 0].plot(max_drawdowns, label='Max Drawdown', color='red')
    axs[1, 0].set_title(f'{symbol} Max Drawdown')
    axs[1, 0].set_ylabel('Max Drawdown')
    axs[1, 0].legend()

    # Risk to Reward Ratio
    axs[1, 1].plot(risk_to_reward_ratios, label='Risk to Reward Ratio', color='green')
    axs[1, 1].set_title(f'{symbol} Risk to Reward Ratio')
    axs[1, 1].set_ylabel('Risk to Reward Ratio')
    axs[1, 1].set_xlabel('Trade Number')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path + '/metrics.png')  # Save the figure to a file
    plt.close()  # Close the figure to free up memory
        
def get_trade_metrics(trade_history, symbols):
    metrics = {}
    for i, symbol in enumerate(symbols):
        trades = [trade for trade in trade_history if trade['symbol'] == symbol]
        num_trades = len(trades)
        wins = [trade for trade in trades if trade['pnl'] > 0]
        losses = [trade for trade in trades if trade['pnl'] <= 0]
        win_rate = len(wins) / num_trades if num_trades > 0 else 0
        avg_profit = np.mean([trade['pnl'] for trade in wins]) if wins else 0
        avg_loss = np.mean([trade['pnl'] for trade in losses]) if losses else 0
        returns = [trade['pnl'] / trade['collateral'] for trade in trades]
        profits = [trade['pnl'] for trade in trades]
        
        # Count TPs, SLs, and liquidations
        num_tps = sum(1 for trade in trades if trade.get('exit_reason') == 'tp')
        num_sls = sum(1 for trade in trades if trade.get('exit_reason') == 'sl')
        num_liqs = sum(1 for trade in trades if trade.get('exit_reason') == 'liq')
        
        metrics[symbol] = {
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'net_profit': net_profit(profits),  # Calculate net profit using returns
            'sharpe_ratio': sharpe_ratio(returns),  # Calculate Sharpe ratio using returns
            'max_drawdown': max_drawdown(returns),  # Calculate max drawdown using cumulative returns
            'risk_to_reward_ratio': risk_to_reward_ratio(returns),  # Calculate risk to reward ratio using returns
            'num_tps': num_tps,  # Number of take profits
            'num_sls': num_sls,  # Number of stop losses
            'num_liqs': num_liqs  # Number of liquidations
        }
        
    # Calculate portfolio metrics
    num_trades = len(trade_history)
    wins = [trade for trade in trade_history if trade['pnl'] > 0]
    losses = [trade for trade in trade_history if trade['pnl'] <= 0]
    win_rate = len(wins) / num_trades if num_trades > 0 else 0
    avg_profit = np.mean([trade['pnl'] for trade in wins]) if wins else 0
    avg_loss = np.mean([trade['pnl'] for trade in losses]) if losses else 0
    returns = [trade['pnl'] / trade['collateral'] for trade in trade_history]
    profits = [trade['pnl'] for trade in trade_history]
    
    # Count TPs, SLs, and liquidations for the portfolio
    num_tps = sum(1 for trade in trade_history if trade.get('exit_reason') == 'tp')
    num_sls = sum(1 for trade in trade_history if trade.get('exit_reason') == 'sl')
    num_liqs = sum(1 for trade in trade_history if trade.get('exit_reason') == 'liq')
    
    metrics['PF'] = {
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'net_profit': net_profit(profits),  # Calculate net profit using returns
        'sharpe_ratio': sharpe_ratio(returns),  # Calculate Sharpe ratio using returns
        'max_drawdown': max_drawdown(returns),  # Calculate max drawdown using cumulative returns
        'risk_to_reward_ratio': risk_to_reward_ratio(returns),  # Calculate risk to reward ratio using returns
        'num_tps': num_tps,  # Number of take profits
        'num_sls': num_sls,  # Number of stop losses
        'num_liqs': num_liqs  # Number of liquidations
    }
    
    # logging.debug(pformat(trade_history))  # Pretty print trade history
    return metrics

def display_trade_metrics(env):
    headers = ["Crypto", "Num Trades", "Win Rate (%)", "Avg Profit", "Avg Loss", "Net Profit", "Sharpe Ratio", "Max Drawdown", "Risk to Reward", "Num Tps", "Num Sls", "Num Liqs"]
    table = []
    
    metrics = get_trade_metrics(env.trade_history, env.symbols)
    for symbol, metric in metrics.items():
        if symbol != 'PF':
            table.append([
                symbol, 
                metric['num_trades'], 
                f"{metric['win_rate'] * 100:.2f}",  # Win rate as percentage with 2 decimals
                f"{metric['avg_profit']:,.2f}",  # Avg profit with 2 decimals and thousands separators
                f"{metric['avg_loss']:,.2f}",  # Avg loss with 2 decimals and thousands separators
                f"{metric['net_profit']:,.2f}",  # Net profit with 2 decimals and thousands separators
                f"{metric['sharpe_ratio']:.3f}",  # Sharpe ratio with 3 decimals
                f"{metric['max_drawdown']:.3f}",  # Max drawdown with 3 decimals
                f"{metric['risk_to_reward_ratio']:.3f}",  # Risk to reward ratio with 3 decimals
                metric['num_tps'],
                metric['num_sls'], 
                metric['num_liqs'], 
            ])
    
    colalign = ("left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right")
    print(tabulate(table, headers=headers, tablefmt="pretty", colalign=colalign))
    
    # Display portfolio metrics
    portfolio_metrics = metrics.get('PF', {})
    if portfolio_metrics:
        portfolio_table = [[
            'PF', 
            portfolio_metrics['num_trades'], 
            f"{portfolio_metrics['win_rate'] * 100:.2f}", 
            f"{portfolio_metrics['avg_profit']:,.2f}", 
            f"{portfolio_metrics['avg_loss']:,.2f}", 
            f"{portfolio_metrics['net_profit']:,.2f}", 
            f"{portfolio_metrics['sharpe_ratio']:.3f}", 
            f"{portfolio_metrics['max_drawdown']:.3f}", 
            f"{portfolio_metrics['risk_to_reward_ratio']:.3f}",
            portfolio_metrics['num_tps'],
            portfolio_metrics['num_sls'], 
            portfolio_metrics['num_liqs'], 
        ]]
        print(tabulate(portfolio_table, headers=headers, tablefmt="pretty", colalign=colalign))

def convert_matrix_to_dict(stock_data, symbols, mapping, timestamps):
    # Ensure matrix is a 3D numpy array
    if not isinstance(stock_data, np.ndarray) or stock_data.ndim != 3:
        raise ValueError("The matrix must be a 3D numpy array.")
    
    # Extract the symbols from the second dimension
    num_symbols = stock_data.shape[1]
    symbols = [symbols[i] for i in range(num_symbols)]
    
    # Ensure the length of timestamps matches the number of rows in the matrix
    num_rows = stock_data.shape[0]
    if len(timestamps) != num_rows:
        raise ValueError(f"Length of timestamps ({len(timestamps)}) does not match number of rows in the matrix ({num_rows}).")
    
    # Create a dictionary to hold the data for each symbol
    result = {}
    
    for i, symbol in enumerate(symbols):
        # Use mapping to select the required columns
        symbol_data = stock_data[:, i, [mapping['volume'], mapping['high'], mapping['low'], mapping['open'], mapping['close']]]
        df = pd.DataFrame(symbol_data, columns=['volume', 'high', 'low', 'open', 'close'])
        
        # Convert provided timestamps to datetime
        df['timestamp'] = pd.to_datetime(timestamps[:num_rows])  # Removed unit='s'
        
        result[symbol] = df
    
    return result

def plot_symbol(symbol, symbol_data, trade_history, title, save_path):
    start_date = symbol_data['timestamp'].iloc[0].strftime('%Y-%m-%d')
    end_date = symbol_data['timestamp'].iloc[-1].strftime('%Y-%m-%d')

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(15, 10))
    
    close = symbol_data['close'].values
    open = symbol_data['open'].values
    high = symbol_data['high'].values
    low = symbol_data['low'].values
    timestamp = symbol_data['timestamp'].values

    up = close >= open
    down = ~up

    width = 0.6 * (timestamp[1] - timestamp[0]) / np.timedelta64(1, 'D')
    width2 = 0.2 * width

    ax1.bar(timestamp[up], close[up]-open[up], width, bottom=open[up], color='green', edgecolor='black')
    ax1.bar(timestamp[up], high[up]-close[up], width2, bottom=close[up], color='green', edgecolor='black')
    ax1.bar(timestamp[up], low[up]-open[up], width2, bottom=open[up], color='green', edgecolor='black')
    ax1.bar(timestamp[down], close[down]-open[down], width, bottom=open[down], color='red', edgecolor='black')
    ax1.bar(timestamp[down], high[down]-open[down], width2, bottom=open[down], color='red', edgecolor='black')
    ax1.bar(timestamp[down], low[down]-close[down], width2, bottom=close[down], color='red', edgecolor='black')

    if trade_history is not None:
        symbol_trades = trade_history[trade_history['symbol'] == symbol].copy()
        symbol_trades['timestamp'] = pd.to_datetime(symbol_trades['timestamp'], unit='ms')
        buys = symbol_trades[symbol_trades['type'] == 'long']
        sells = symbol_trades[symbol_trades['type'] == 'short']

        y_shift = 0.05
        for _, trade in buys.iterrows():
            trade_time = trade['timestamp']
            trade_price = low[np.where(timestamp == trade_time)[0][0]] * (1 - y_shift)
            ax1.plot([trade_time, trade_time], [trade_price, low[np.where(timestamp == trade_time)[0][0]]], 
                     color='g', linewidth=1, linestyle='-')
            ax1.scatter(trade_time, trade_price, marker='^', color='g', s=100)

        for _, trade in sells.iterrows():
            trade_time = trade['timestamp']
            trade_price = high[np.where(timestamp == trade_time)[0][0]] * (1 + y_shift)
            ax1.plot([trade_time, trade_time], [trade_price, high[np.where(timestamp == trade_time)[0][0]]], 
                     color='r', linewidth=1, linestyle='-')
            ax1.scatter(trade_time, trade_price, marker='v', color='r', s=100)

    volume_width = 0.6 * width
    ax2.bar(timestamp, symbol_data['volume'], width=volume_width, color=np.where(close >= open, 'green', 'red'), alpha=0.6)

    ax1.set_xticklabels([])
    ax1.set_title(f"{title} ({symbol})\nPeriod: {start_date} to {end_date}")
    ax1.set_ylabel("Price")
    ax2.set_ylabel("Volume")

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    
    if save_path:
        events_output_dir = os.path.join(save_path, 'events')
        os.makedirs(events_output_dir, exist_ok=True)
        plt.savefig(os.path.join(events_output_dir, f'{symbol}.png'))
    else:
        plt.show()

    plt.close()

def plot_stock_data(stock_data, symbols, mapping, timestamps, trade_history=None, title="Candlestick Chart with Volume", save_path=None):
    stock_data_dict = convert_matrix_to_dict(stock_data, symbols, mapping, timestamps)

    start_time = time.time()
    print('plot_stock_data started')

    if trade_history is not None:
        trade_history = pd.DataFrame(trade_history)

    # Use multiprocessing to plot symbols in parallel
    with ProcessPoolExecutor(initializer=suppress_urllib3_warning) as executor:
        futures = [executor.submit(plot_symbol, symbol, data, trade_history, title, save_path) 
                   for symbol, data in stock_data_dict.items()]
        for future in as_completed(futures):
            future.result()  # This will raise any exceptions that occurred during execution

    print('plot_stock_data stopped')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_cumulative_returns(trade_history, test_data, mapping, valid_symbols, save_path='cumulative_returns.png'):
    # Convert trade history to DataFrame if it's not already
    if not isinstance(trade_history, pd.DataFrame):
        trade_history = pd.DataFrame(trade_history)
    
    # Ensure trade_history has the necessary columns
    required_columns = ['timestamp', 'pnl', 'collateral']
    if not all(col in trade_history.columns for col in required_columns):
        raise ValueError(f"trade_history must have columns: {required_columns}")
    
    # Convert timestamp to datetime if it's not already
    trade_history['timestamp'] = pd.to_datetime(trade_history['timestamp'], unit='ms')
    
    # Sort trade history by timestamp
    trade_history = trade_history.sort_values('timestamp')
    
    # Calculate returns as pnl / collateral
    trade_history['return'] = trade_history['pnl'] / trade_history['collateral']
    
    # Calculate cumulative returns using cumsum
    trade_history['cumulative_return'] = trade_history['return'].cumsum()
    
    # The final return is the last value in the cumulative return series
    strategy_final_return = trade_history['cumulative_return'].iloc[-1] / 100

    # Calculate buy and hold returns
    start_prices = test_data[0, :, mapping['close']]
    end_prices = test_data[-1, :, mapping['close']]
    buy_hold_returns = (end_prices - start_prices) / start_prices
    
    # Calculate the average buy and hold return across all symbols
    avg_buy_hold_return = np.mean(buy_hold_returns)

    # Create a more realistic buy and hold strategy
    buy_hold_df = pd.DataFrame({
        'timestamp': pd.date_range(start=trade_history['timestamp'].iloc[0], 
                                   end=trade_history['timestamp'].iloc[-1], 
                                   periods=len(test_data) - 1)  # Adjust periods to match daily_returns
    })

    # Initialize daily_returns with one less row than test_data
    daily_returns = np.zeros((len(test_data) - 1, len(valid_symbols)))

    for i, symbol in enumerate(valid_symbols):
        # Calculate the difference between consecutive close prices
        price_diff = np.diff(test_data[:, i, mapping['close']])
        
        # Use the previous close prices for the division
        prev_close_prices = test_data[:-1, i, mapping['close']]
        
        # Calculate daily returns as (next close - prev close) / prev close
        daily_returns[:, i] = price_diff / prev_close_prices

    # Calculate the average daily return across all symbols
    avg_daily_returns = np.mean(daily_returns, axis=1)

    # Multiply by 100 to convert to percentage
    avg_daily_returns *= 100

    # Debug: Print avg_daily_returns
    # logging.info(f"avg_daily_returns: {avg_daily_returns}")

    # Calculate the cumulative returns for the buy and hold strategy using cumsum
    buy_hold_df['cumulative_return'] = np.cumsum(avg_daily_returns)
    
    # # Debug: Print cumulative returns
    # logging.info(f"buy_hold_df['cumulative_return']: {buy_hold_df['cumulative_return']}")
    
    # logging.info(f"avg_daily_returns: {avg_daily_returns}")
    # logging.info(f"daily_returns: {daily_returns}")

    # Plotting
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])
    
    # Plot cumulative returns
    ax1.plot(trade_history['timestamp'], trade_history['cumulative_return'], 
            label='Trading Strategy', color='blue')
    ax1.plot(buy_hold_df['timestamp'], buy_hold_df['cumulative_return'], 
            label='Buy and Hold', color='red')
    
    ax1.set_ylabel('Cumulative Returns')
    ax1.set_title('Trading Strategy vs Buy and Hold Cumulative Returns')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot individual returns
    ax2.scatter(trade_history['timestamp'], trade_history['return'], 
                label='Individual Trade Returns', color='green', alpha=0.5, s=10)
    ax2.axhline(y=0, color='r', linestyle='--')
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Individual Returns')
    ax2.set_title('Individual Trade Returns')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Format x-axis for ax1 and ax2
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot distribution of returns
    sns.histplot(trade_history['return'], kde=True, ax=ax3, color='purple')
    ax3.set_xlabel('Return')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Returns')
    ax3.axvline(x=0, color='r', linestyle='--')
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path)
    plt.close()

    # Print the final returns
    print(f"Trading Strategy Final Return: {strategy_final_return:.2%}")
    print(f"Buy and Hold Final Return: {avg_buy_hold_return:.2%}")

    # Additional analysis
    print(f"Total trades: {len(trade_history)}")
    print(f"Average return per trade: {trade_history['return'].mean():.4%}")
    print(f"Median return per trade: {trade_history['return'].median():.4%}")
    print(f"Standard deviation of returns: {trade_history['return'].std():.4%}")
    print(f"Percentage of profitable trades: {(trade_history['return'] > 0).mean():.2%}")
    print(f"Largest gain: {trade_history['return'].max():.2%}")
    print(f"Largest loss: {trade_history['return'].min():.2%}")
    
    # Calculate Sharpe Ratio (assuming risk-free rate of 0 for simplicity)
    sharpe_ratio = trade_history['return'].mean() / trade_history['return'].std() * np.sqrt(252)  # Annualized
    
    # Calculate Maximum Drawdown
    cumulative_returns = trade_history['cumulative_return']
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min() / 100
    
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")

    # print(f"Top 5 largest losses:")
    # print(trade_history.nsmallest(5, 'return')[['timestamp', 'symbol', 'return']])
    # print(f"Top 5 largest gains:")
    # print(trade_history.nlargest(5, 'return')[['timestamp', 'symbol', 'return']])
    
    print(f"Top 5 largest losses:")
    print(trade_history.nsmallest(5, 'return'))  # Display all fields
    print(f"Top 5 largest gains:")
    print(trade_history.nlargest(5, 'return'))  # Display all fields

def plot_sl_tp_distribution(distribution, save_path='sl_tp_distribution.png'):
    methods = ['sl_fractal', 'sl_atr', 'sl_percentage', 'tp_fractal', 'tp_atr', 'tp_percentage']
    counts = [distribution[method] for method in methods]
    total = sum(counts)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=methods, y=counts, ax=ax, hue=methods, palette='viridis', legend=False)

    ax.set_title('Distribution of SL and TP Methods')
    ax.set_xlabel('Method')
    ax.set_ylabel('Count')
    ax.set_yscale('log')  # Set y-axis to log scale for better visualization

    # Annotate each bar with the percentage
    for i, count in enumerate(counts):
        percentage = (count / total) * 100
        ax.text(i, count, f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10, color='black')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

import unittest

class TestConvertMatrixToDict(unittest.TestCase):
    def setUp(self):
        # Create a sample input matrix (num_candles x num_symbols x num_features)
        self.num_candles = 1000
        self.num_symbols = 2
        self.num_features = 15  # Number of features in the mapping
        self.symbols = ['BTC', 'ETH']
        self.mapping = {
            'atr': 0, 'boll_hband': 1, 'boll_lband': 2, 'boll_mband': 3, 'close': 4, 
            'ema_long': 5, 'ema_short': 6, 'high': 7, 'low': 8, 'macd': 9, 
            'macd_hist': 10, 'macd_signal': 11, 'open': 12, 'rsi': 13, 'volume': 14
        }

        # Create a sample matrix with random data
        self.matrix = np.random.rand(self.num_candles, self.num_symbols, self.num_features)
        
        # Create a sample timestamps list
        self.timestamps = np.arange(1, self.num_candles + 1) * 1000000  # Example timestamps in seconds

    def test_convert_matrix_to_dict(self):
        result = convert_matrix_to_dict(self.matrix, self.symbols, self.mapping, self.timestamps)
        
        # Check if the result is a dictionary
        self.assertIsInstance(result, dict)
        
        # Check if the dictionary has the correct symbols
        self.assertListEqual(list(result.keys()), self.symbols)
        
        for symbol in self.symbols:
            df = result[symbol]
            
            # Check if the DataFrame has the correct columns
            expected_columns = ['volume', 'high', 'low', 'open', 'close', 'timestamp']
            self.assertListEqual(list(df.columns), expected_columns)
            
            # Check if the DataFrame has the correct number of rows
            self.assertEqual(len(df), self.num_candles)
            
            # Check if the 'timestamp' column is correctly converted to datetime
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['timestamp']))


if __name__ == '__main__':
    unittest.main()