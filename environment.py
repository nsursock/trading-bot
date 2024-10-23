import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
# import time
from pprint import pformat
from utilities import calculate_leverage, compute_liquidation_price, max_drawdown

# Set up logging
# logging.basicConfig(level=logging.DEBUG)

class CryptoTradingEnv(gym.Env):
    def __init__(self, data, mapping, symbols, timestamps, params):
        # Ensure data is a NumPy array of floats
        self.data = np.array(data, dtype=np.float32)
        self.params = params
        
        # Check if data has the expected shape
        if len(self.data.shape) != 3:
            raise ValueError(f"Expected data to have 3 dimensions, but got {self.data.shape}")
        self.mapping = mapping
        self.symbols = sorted(symbols)
        self.timestamps = timestamps
        self.initial_balance = params['initial_balance']
        self.risk_per_trade = params['risk_per_trade']
        self.min_collateral = params['min_collateral']
        self.current_step = 0
        self.balance = params['initial_balance']
        self.positions = [{} for _ in range(self.data.shape[1])]  # Initialize positions as a list of dictionaries
        self.trade_history = []  # Initialize trade history
        self.action_space = spaces.MultiDiscrete([4] * self.data.shape[1])  # 0: Hold, 1: Buy (Long), 2: Sell (Short), 3: Close
        
        # Update the observation space to match the actual observation shape
        obs_shape = (self.data.shape[1] * self.data.shape[2] + 1 + self.data.shape[1],)  # Added +1 for regime
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        
        self.cooldown_period = params['cooldown_period']
        self.cooldown_counter = 0  # Initialize cooldown counter
        
        self.rewards = []
        self.actions = []
        
        self.total_wins = 0
        self.total_win_pnl = 0.0
        self.total_loss_pnl = 0.0
        
        self.tp_atr_mult = params['tp_atr_mult'] or 4
        self.sl_atr_mult = params['sl_atr_mult'] or 2
        self.tp_percentage = params['tp_percentage'] or 0.1  # Default 5%
        self.sl_percentage = params['sl_percentage'] or 0.02  # Default 1%
        
        # print(f"Data shape: {self.data.shape}")
        
        # Initialize counters for SL and TP methods
        self.sl_fractal_count = 0
        self.sl_atr_count = 0
        self.sl_percentage_count = 0
        self.tp_fractal_count = 0
        self.tp_atr_count = 0
        self.tp_percentage_count = 0
        
    def update_data(self, data, timestamps):
        self.data = data
        self.timestamps = timestamps
        
    def reset(self, seed=None):
        super().reset(seed=seed)  # Ensure the parent class reset is called with the seed
        self.current_step = 0
        self.balance = self.initial_balance
        self.positions = [{} for _ in range(self.data.shape[1])]  # Initialize positions as a list of dictionaries
        self.cooldown_counter = 0  # Reset cooldown counter
        return self._get_observation(), {}


    def _handle_tp_sl_liq(self, current_prices):
        """
        Handle take profit, stop loss, and liquidation for all positions.
        """
        for i, position in enumerate(self.positions):
            if position:
                tp_price = position['tp_price']
                sl_price = position['sl_price']
                liq_price = position['liq_price']
                
                if position['type'] == 'long':
                    if np.all(current_prices[i] >= tp_price):
                        self._close_position(i, current_prices[i], 'tp')
                    elif np.all(current_prices[i] <= liq_price):
                        self._close_position(i, current_prices[i], 'liq')
                    elif np.all(current_prices[i] <= sl_price):
                        self._close_position(i, current_prices[i], 'sl')
                elif position['type'] == 'short':
                    if np.all(current_prices[i] <= tp_price):
                        self._close_position(i, current_prices[i], 'tp')
                    elif np.all(current_prices[i] >= liq_price):
                        self._close_position(i, current_prices[i], 'liq')
                    elif np.all(current_prices[i] >= sl_price):
                        self._close_position(i, current_prices[i], 'sl')

    def _close_position(self, index, current_price, reason='other'):
        position = self.positions[index]
        if position['type'] == 'long':
            pnl = (current_price - position['entry_price']) * position['size']
        elif position['type'] == 'short':
            pnl = (position['entry_price'] - current_price) * position['size']
        
        self.balance += position['collateral'] + pnl
        position['exit_price'] = current_price
        position['pnl'] = pnl
        position['exit_reason'] = reason
        
        self.trade_history.append(position)
        
         # Update running totals
        if pnl > 0:
            self.total_wins += 1
            self.total_win_pnl += pnl
        else:
            self.total_loss_pnl -= pnl
        
        self.positions[index] = {}
        

    def _get_returns_std(self, index):
        """
        Calculate the standard deviation of returns for a given symbol.
        """
        returns = np.diff(self.data[:, index, self.mapping['close']]) / self.data[:-1, index, self.mapping['close']]
        return np.std(returns)
    
    
    def calculate_var(self, asset_returns, confidence_level=0.95):
        if not (0 <= confidence_level <= 1):
            raise ValueError("Confidence level must be between 0 and 1")
        var = np.percentile(asset_returns, (1 - confidence_level) * 100)
        return var

    def dynamic_leverage(self, target_var, current_var):
        leverage = target_var / abs(current_var)
        return leverage
    
    def adjust_leverage(self, base_leverage, params):
        interval_risk_factors = {
            '1m': 6.0,   # High leverage for short intervals
            '3m': 5.0,
            '5m': 4.0,
            '15m': 3.5,
            '30m': 3.0,
            '1h': 2.5,
            '2h': 2.0,
            '4h': 1.8,
            '6h': 1.5,
            '8h': 1.3,
            '12h': 1.2,
            '1d': 1.1,
            '3d': 1.0,   # Base leverage for long intervals
            '1w': 0.9,
            '1M': 0.8    # Lowest leverage for very long intervals
        }
        
        interval = params['interval']

        # Ensure the interval exists in the dictionary
        if interval not in interval_risk_factors:
            logging.warning(f"Interval {interval} is not supported.")
            return 1

        risk_factor = interval_risk_factors[interval]
        adjusted_leverage = base_leverage * risk_factor  # Decreasing leverage for longer intervals
        
        # Clamp the adjusted leverage between 1 and 150
        adjusted_leverage = max(params['leverage_min'], min(adjusted_leverage, params['leverage_max']))  # Ensure leverage is between 1 and 150

        return adjusted_leverage
        
    def _open_position(self, i, current_price, direction):
        # Close existing position if it exists
        if self.positions[i]:
            self._close_position(i, current_price, 'opening')
            
        leverage = 1  # Default leverage if conditions are not met
        risk_amount = round(self.balance * self.risk_per_trade)
        tp_price = current_price * 9  # Set TP price to 900% of the current price
        sl_price = compute_liquidation_price(current_price, leverage, direction)  # Set SL price to liquidation price
        
        if self.balance > self.min_collateral and current_price != 0:
            leverage = calculate_leverage(self.data[:, i, :], self.mapping, risk_amount)[self.current_step]
            if self.params['boost_leverage']:
                leverage = self.adjust_leverage(leverage, self.params)
            
            # sl_price = compute_liquidation_price(current_price, leverage, direction)  # Set SL price to liquidation price
            
            #  # Calculate asset returns
            # asset_returns = np.diff(self.data[:, i, self.mapping['close']]) / self.data[:-1, i, self.mapping['close']]
            # current_var = self.calculate_var(asset_returns, self.params['confidence_level'])
            # target_var = self.params['target_var']  # Example target VaR (5%)
            # leverage = self.dynamic_leverage(target_var, current_var)
            # 
            # # Ensure leverage is between leverage_min and leverage_max
            # leverage = max(self.params['leverage_min'], min(leverage, self.params['leverage_max']))
            
            
            
            # calculate leverage based on VaR
            
            # win_probability = self._estimate_win_probability()
            # win_loss_ratio = self._estimate_win_loss_ratio()
            # returns_std = self._get_returns_std(i)
            
            # if returns_std > 0:
            #     kelly_leverage = (win_probability - (1 - win_probability) / win_loss_ratio) / (returns_std ** 2)
            #     # print('>>>>>>>>>>>>>>>', kelly_leverage)
            #     leverage = max(params['leverage_min'], min(kelly_leverage, params['leverage_max']))  # Ensure leverage is between 1 and 150
            # else:
            #     leverage = params['leverage_min']  # Default leverage if returns_std is zero
        
            
            position_size = (risk_amount * leverage) / current_price
            # atr_value = self.data[self.current_step, i, self.mapping['atr']]
            tp_price, sl_price, liq_price = self._compute_prices(current_price, direction, i, leverage)
            self.positions[i] = {
                'timestamp': self.timestamps[self.current_step],  # Access the timestamp assuming it's the first dimension
                'entry_price': current_price,
                'size': position_size,
                'leverage': leverage,
                'collateral': risk_amount,
                'type': direction,
                'symbol': self.symbols[i],
                'tp_price': tp_price,
                'sl_price': sl_price,
                'liq_price': liq_price,
                'pnl': 0.0  # Initialize PnL
            }
            self.balance -= risk_amount
            self.cooldown_counter = self.cooldown_period  # Set cooldown
            
        return risk_amount, leverage, tp_price, sl_price
            
    def _get_fractal_high(self, current_step, symbol_index):
        if current_step < 2 or current_step > len(self.data) - 3:
            return None
        window = self.data[current_step-2:current_step+3, symbol_index, self.mapping['high']]
        if len(window) == 5 and window[2] == np.max(window):
            return window[2]
        return None

    def _get_fractal_low(self, current_step, symbol_index):
        if current_step < 2 or current_step > len(self.data) - 3:
            return None
        window = self.data[current_step-2:current_step+3, symbol_index, self.mapping['low']]
        if len(window) == 5 and window[2] == np.min(window):
            return window[2]
        return None
    
    def _compute_sl(self, liq_price, entry_price, direction, symbol_index, fractal_high, fractal_low):
        # Attempt to use fractals
        fractal = fractal_low if direction == 'long' else fractal_high
        if fractal is not None:
            logging.debug(f"Symbol {symbol_index} will use fractal for SL")
            self.sl_fractal_count += 1
            return fractal
        
        # Attempt to use ATR as fallback
        atr_value = self.data[self.current_step, symbol_index, self.mapping['atr']]
        if direction == 'long':
            sl = entry_price - self.sl_atr_mult * atr_value
            if sl < entry_price and sl > liq_price:
                logging.debug(f"Symbol {symbol_index} will use ATR for SL long: ({entry_price, atr_value})")
                self.sl_atr_count += 1
                return sl
        elif direction == 'short':
            sl = entry_price + self.tp_atr_mult * atr_value
            if sl > entry_price and sl < liq_price:
                logging.debug(f"Symbol {symbol_index} will use ATR for SL short: ({entry_price, atr_value})")
                self.sl_atr_count += 1
                return sl
            
        # Use percentage as final fallback
        logging.debug(f"Symbol {symbol_index} will use percentage for SL")
        self.sl_percentage_count += 1
        sl_short = entry_price * (1 + self.tp_percentage) or 0
        sl_long = entry_price * (1 - self.sl_percentage) or 0
        return sl_long if direction == 'long' else sl_short 

    def _compute_tp(self, entry_price, direction, symbol_index, fractal_high, fractal_low):
        # Attempt to use fractals
        fractal = fractal_high if direction == 'long' else fractal_low
        if fractal is not None:
            logging.debug(f"Symbol {symbol_index} will use fractal for TP")
            self.tp_fractal_count += 1
            return fractal

        # Attempt to use ATR as fallback
        atr_value = self.data[self.current_step, symbol_index, self.mapping['atr']]
        if direction == 'long':
            tp = entry_price + self.tp_atr_mult * atr_value
            if tp > entry_price:
                logging.debug(f"Symbol {symbol_index} will use ATR for TP long: ({entry_price, atr_value})")
                self.tp_atr_count += 1
                return tp
        else:
            tp = entry_price - self.tp_atr_mult * atr_value
            if tp < entry_price:
                logging.debug(f"Symbol {symbol_index} will use ATR for TP short: ({entry_price, atr_value})")
                self.tp_atr_count += 1
                return tp
            
        # Use percentage as final fallback
        logging.debug(f"Symbol {symbol_index} will use percentage for TP")
        self.tp_percentage_count += 1
        tp_short = entry_price * (1 - self.tp_percentage) or 0
        tp_long = entry_price * (1 + self.sl_percentage) or 0
        return tp_long if direction == 'long' else tp_short 

    # def _compute_sl(self, entry_price, direction, symbol):
    #     if direction == 'long':
    #         for step in range(self.current_step, -1, -1):
    #             fractal_low = self._get_fractal_low(step, symbol)
    #             if fractal_low is not None and fractal_low < entry_price:
    #                 return fractal_low
    #         # Use ATR as fallback
    #         logging.info(f"SL Long: Using ATR, could not find fractals (symbol: {symbol})")
    #         atr_value = self.data[self.current_step, symbol, self.mapping['atr']]  # Example: mean ATR of all symbols
    #         return entry_price - self.sl_atr_mult * atr_value  # Adjust SL below entry for long positions
    #     else:  # short
    #         for step in range(self.current_step, -1, -1):
    #             fractal_high = self._get_fractal_high(step, symbol)
    #             if fractal_high is not None and fractal_high > entry_price:
    #                 return fractal_high
    #         # Use ATR as fallback
    #         logging.info(f"SL Short: Using ATR, could not find fractals (symbol: {symbol})")
    #         atr_value = self.data[self.current_step, symbol, self.mapping['atr']]  # Example: mean ATR of all symbols
    #         return entry_price + self.sl_atr_mult * atr_value  # Adjust SL above entry for short positions

    # def _compute_tp(self, entry_price, direction, symbol):
    #     if direction == 'long':
    #         for step in range(self.current_step, -1, -1):
    #             fractal_high = self._get_fractal_high(step, symbol)
    #             if fractal_high is not None and fractal_high > entry_price:
    #                 return fractal_high
    #         # Use ATR as fallback
    #         logging.info(f"TP Long: Using ATR, could not find fractals (symbol: {symbol})")
    #         atr_value = self.data[self.current_step, symbol, self.mapping['atr']]  # Example: mean ATR of all symbols
    #         return entry_price + self.tp_atr_mult * atr_value  # Adjust TP above entry for long positions
    #     else:  # short
    #         for step in range(self.current_step, -1, -1):
    #             fractal_low = self._get_fractal_low(step, symbol)
    #             if fractal_low is not None and fractal_low < entry_price:
    #                 return fractal_low
    #         # Use ATR as fallback
    #         logging.info(f"TP Short: Using ATR, could not find fractals (symbol: {symbol})")
    #         atr_value = self.data[self.current_step, symbol, self.mapping['atr']]  # Example: mean ATR of all symbols
    #         return entry_price - self.tp_atr_mult * atr_value  # Adjust TP below entry for short positions

    def _get_fractals(self, symbol_index, entry_price):
        for step in range(self.current_step, -1, -1):
            fractal_high = self._get_fractal_high(step, symbol_index)
            fractal_low = self._get_fractal_low(step, symbol_index)
            if fractal_low is not None and fractal_low < entry_price and fractal_high is not None and fractal_high > entry_price:
                return fractal_high, fractal_low
        return None, None
        
    def _compute_prices(self, current_price, direction, symbol_index, leverage):
        fractal_high, fractal_low = self._get_fractals(symbol_index, current_price)
        tp_price = self._compute_tp(current_price, direction, symbol_index, fractal_high, fractal_low)
        liq_price = compute_liquidation_price(current_price, leverage, direction)
        sl_price = self._compute_sl(liq_price, current_price, direction, symbol_index, fractal_high, fractal_low)
        
        return tp_price, sl_price, liq_price

    def step(self, action):
        logging.debug(f"Step called with action: {action} (type: {type(action)})")
        
        if self.current_step >= len(self.data) - 1:
            raise IndexError("Current step exceeds data length.")

        current_prices = self.data[self.current_step, :, self.mapping['close']]
        # Ensure current_prices is not empty
        if current_prices.size == 0:
            raise ValueError("Current prices array is empty.")
        
        infos = {}
        
        self._handle_tp_sl_liq(current_prices)  # Handle TP, SL, and liquidation before processing new actions
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1  # Decrement cooldown counter
            action = [act if act == 0 or act == 3 else 0 for act in action]  # Override actions to hold if cooldown is active
            # print(f"Action overridden due to cooldown: {action}")  # Debug print for cooldown
        
        for i, act in enumerate(action):
            if act == 1:  # Buy (Long)
                # Ensure index i is within bounds for current_prices
                if i >= len(current_prices):
                    raise IndexError(f"Index {i} out of bounds for current_prices with length {len(current_prices)}.")
                collateral, leverage, tp_price, sl_price = self._open_position(i, current_prices[i], 'long')
                infos[self.symbols[i]] = { 'type': 'open_long', 'collateral': collateral, 'leverage': leverage, 'tp_price': tp_price, 'sl_price': sl_price }
                # print(f"Buy (Long) action taken for {self.symbols[i]}: {infos[self.symbols[i]]}")
        
            elif act == 2:  # Sell (Short)
                if i >= len(current_prices):
                    raise IndexError(f"Index {i} out of bounds for current_prices with length {len(current_prices)}.")
                collateral, leverage, tp_price, sl_price = self._open_position(i, current_prices[i], 'short')
                infos[self.symbols[i]] = { 'type': 'open_short', 'collateral': collateral, 'leverage': leverage, 'tp_price': tp_price, 'sl_price': sl_price }
                # print(f"Sell (Short) action taken for {self.symbols[i]}: {infos[self.symbols[i]]}")
        
            elif act == 3:  # Close
                if self.positions[i]:
                    if i >= len(current_prices):
                        raise IndexError(f"Index {i} out of bounds for current_prices with length {len(current_prices)}.")
                    self._close_position(i, current_prices[i])  
                    infos[self.symbols[i]] = { 'type': 'close', 'exit_price': current_prices[i] }
                    # print(f"Close action taken for {self.symbols[i]}")
    
        # Update PnL for all open positions
        for i, position in enumerate(self.positions):
            if position:
                if position['type'] == 'long':
                    position['pnl'] = (current_prices[i] - position['entry_price']) * position['size']
                elif position['type'] == 'short':
                    position['pnl'] = (position['entry_price'] - current_prices[i]) * position['size']
    
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1 or self.balance < self.min_collateral
        next_state = self._get_observation()
        
        # Calculate reward
        # profit = self.balance + sum(pos.get('pnl', 0) for pos in self.positions if pos) - self.initial_balance
        # risk = np.std([self.balance] + [pos.get('pnl', 0) for pos in self.positions if pos])
        # reward = profit - risk  # Maximize profit and minimize risk
        reward = self._calculate_reward(action)
        
        self.rewards.append(reward)
        self.actions.append(action)
        
        # Update risk per trade using fractional Kelly criterion
        self._update_risk_per_trade()
        
        # print(f"Infos before returning: {infos}")
        
        if done:
            if self.current_step >= len(self.data) - 1:
                infos['terminal_info'] = 'Max steps reached'
            elif self.balance <= self.min_collateral:
                infos['terminal_info'] = 'Insufficient balance'
            else:
                infos['terminal_info'] = 'Unknown termination reason'

        logging.debug(f"Step {self.current_step}: Balance = {self.balance:.2f}, Risk Per Trade = {self.risk_per_trade:.4f}") #, Portfolio Value = {self.portfolio_value:.2f}")
        
        return next_state, reward, done, False, infos

    def _get_observation(self):
        positions_flat = [pos.get('size', 0) for pos in self.positions]
        obs = np.concatenate((self.data[self.current_step].flatten(), [self.balance], positions_flat))
        # print(f"Observation shape: {obs.shape}")
        return obs
    
    def get_current_observation(self):
        # Logic to get the observation for the current step
        positions_flat = [pos.get('size', 0) for pos in self.positions]
        obs = np.concatenate((self.data[len(self.data) - 1].flatten(), [self.balance], positions_flat))
        # print(f"Observation shape: {obs.shape}")
        return obs
    
    def render(self, mode='human'):
        pass


    def _calculate_reward(self, action):
        current_prices = self.data[self.current_step, :, self.mapping['close']]
        reward = 0
        
        for i, position in enumerate(self.positions):
            if 'leverage' in position and 'collateral' in position:
                position_size = position['leverage'] * position['collateral']
                if position['type'] == 'long':
                    reward += (current_prices[i] - position['entry_price']) * position_size
                elif position['type'] == 'short':
                    reward += (position['entry_price'] - current_prices[i]) * position_size
                
                if action[i] != 0:
                    reward -= self._transaction_cost(i)
        
        # Reward for consecutive positive PnL
        consecutive_positive_pnl = 0
        return_since_open = 0.5
        for i in range(self.data.shape[1]):
            pnl = 0
            for j in range(1, self.current_step + 1):
                if self.positions[i]:
                    if self.positions[i]['type'] == 'long':
                        pnl += (self.data[self.current_step - j, i, self.mapping['close']] - self.positions[i]['entry_price']) * self.positions[i]['size']
                    elif self.positions[i]['type'] == 'short':
                        pnl += (self.positions[i]['entry_price'] - self.data[self.current_step - j, i, self.mapping['close']]) * self.positions[i]['size']
                if pnl > 0:
                    consecutive_positive_pnl += 1
                    return_since_open = pnl / (self.positions[i]['entry_price'] * self.positions[i]['size'])
                else:
                    break
        reward += consecutive_positive_pnl * return_since_open  # Adjust the multiplier as needed
        
        if self.params['normalize_reward']:
            # Normalize
            reward = np.sign(reward) * np.log1p(abs(reward))  # Logarithmic scaling

            # Assuming reward values are between -20,000 and 20,000
            reward_min = -25000
            reward_max = 25000
            
            # Normalize to -1 to 1 range
            reward = 2 * ((reward - reward_min) / (reward_max - reward_min)) - 1
        
        # Penalize for high volatility
        volatility_penalty = self._get_volatility() * 0.1
        reward -= volatility_penalty
        
        # Penalize for large drawdowns using max_drawdown from utils
        max_drawdown_penalty = max_drawdown([self.balance] + [pos.get('pnl', 0) for pos in self.positions if pos]) * 0.5
        reward -= max_drawdown_penalty
        
        # Small penalty for each step to encourage faster decision-making
        step_penalty = 0.01
        reward -= step_penalty
        
        return reward
    

    def _transaction_cost(self, index):
        position = self.positions[index]
        if position:
            position_size = position['collateral'] * position['leverage']
            return max(8, 0.0008 * position_size)
        return 0

    def _get_volatility(self):
        # Define your volatility calculation here
        return np.std(self.data[:, :, self.mapping['close']])
    
    
    def _update_risk_per_trade(self):
        """
        Update the risk per trade using the fractional Kelly criterion,
        handling cases where the full Kelly is negative.
        """
        win_probability = self._estimate_win_probability()
        win_loss_ratio = self._estimate_win_loss_ratio()
        
        if win_loss_ratio > 0:
            full_kelly = win_probability - (1 - win_probability) / win_loss_ratio
            
            # Use a fraction of the full Kelly criterion
            fraction = self.params['kelly_fraction'] or 0.5  # You can adjust this value based on your risk tolerance
            
            if full_kelly > 0:
                fractional_kelly = full_kelly * fraction
                # Ensure the fraction is between 0.01 (1%) and 0.1 (10%)
                self.risk_per_trade = max(0.01, min(fractional_kelly, 0.1))
            else:
                # When full Kelly is negative, use a small, non-zero value
                self.risk_per_trade = max(0.01, min(-full_kelly * fraction * 0.1, 0.05))
        else:
            # If win_loss_ratio is not positive, set a conservative default
            self.risk_per_trade = self.params['risk_per_trade']
        
        # logging.info(f"Updated risk_per_trade: {self.risk_per_trade:.4f}")

    def _estimate_win_probability(self):
        """
        Estimate the probability of winning based on historical trades.
        """
        if not self.trade_history:
            return 0.5  # Default to 50% if no trade history
        
        return self.total_wins / len(self.trade_history)

    def _estimate_win_loss_ratio(self):
        """
        Estimate the win/loss ratio based on historical trades.
        """
        if not self.trade_history:
            return 1  # Default to 1 if no trade history
        
        if self.total_loss_pnl == 0:
            return float('inf')  # Avoid division by zero
        
        return self.total_win_pnl / self.total_loss_pnl

    def get_sl_tp_distribution(self):
        return {
            'sl_fractal': self.sl_fractal_count,
            'sl_atr': self.sl_atr_count,
            'sl_percentage': self.sl_percentage_count,
            'tp_fractal': self.tp_fractal_count,
            'tp_atr': self.tp_atr_count,
            'tp_percentage': self.tp_percentage_count
        }

