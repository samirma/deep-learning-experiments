import numpy as np

class TraderEnv():
    """Class for a discrete (buy/hold/sell) spread trading environment.
    """

    _actions = {
        'hold': 0,
        'buy': 1,
        'sell': 2,
        'cancel_buy': 3,
        'cancel_sell': 4
    }

    _positions = {
        'flat': 0,
        'ordened_sell': 1,
        'ordened_buy': 2,
        'bought': 3
    }
    
    _allowed = {
        _positions['flat']: {'hold', 'buy'},
        _positions['bought']: {'hold', 'sell'},
        _positions['ordened_sell']: {'hold', 'cancel_sell'},
        _positions['ordened_buy']: {'hold', 'cancel_buy'}
    }

    def __init__(self, data_generator, spread_coefficients, episode_length=1000, trading_fee=0, time_fee=0, history_length=2):
        """Initialisation function
        Args:
            data_generator (tgym.core.DataGenerator): A data
                generator object yielding a 1D array of bid-ask prices.
            spread_coefficients (list): A list of signed integers defining
                how much of each product to buy (positive) or sell (negative)
                when buying or selling the spread.
            episode_length (int): number of steps to play the game for
            trading_fee (float): penalty for trading
            time_fee (float): time fee
            history_length (int): number of historical states to stack in the
                observation vector.
        """

        #assert data_generator.n_products == len(spread_coefficients)
        assert history_length > 0
        self._data_generator = data_generator
        self._spread_coefficients = spread_coefficients
        self._first_render = True
        self._trading_fee = trading_fee
        self._time_fee = time_fee
        self._episode_length = episode_length
        self.n_actions = len(_actions)
        self._prices_history = []
        self._history_length = history_length
        self.reset()

    def reset(self):
        """Reset the trading environment. Reset rewards, data generator...
        Returns:
            observation (numpy.array): observation of the state
        """
        self._iteration = 0
        self._data_generator.rewind()
        self._total_reward = 0
        self._total_pnl = 0
        self._position = self._positions['flat']
        self._entry_price = 0
        self._exit_price = 0

        for i in range(self._history_length):
            self._prices_history.append(next(self._data_generator))

        observation = self._get_observation()
        self.state_shape = observation.shape
        self._action = self._actions['hold']
        return observation

    def step(self, action):
        """Take an action (buy/sell/hold) and computes the immediate reward.
        Args:
            action (numpy.array): Action to be taken, one-hot encoded.
        Returns:
            tuple:
                - observation (numpy.array): Agent's observation of the current environment.
                - reward (float) : Amount of reward returned after previous action.
                - done (bool): Whether the episode has ended, in which case further step() calls will return undefined results.
                - info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        
        self._action = action
        self._iteration += 1
        done = False
        instant_pnl = 0
        info = {}
        reward = -self._time_fee
        
        observation = self.get_observation()
        
        if (is_valid(action, self._position):

            if action == self._actions['buy']:
                self.send_order_to_buy()
            elif action == self._actions['sell']:
                self.send_order_to_sell()
            elif action == self._actions['cancel_buy']:
                self.cancel_buy()
            elif action == self._actions['cancel_sell']:
                self.cancel_sell()
            
        else:
            info['status'] = 'Invalid action'
            done = True
            reward = -1000
            
        reward += self.instant_pnl
        self._total_pnl += self.instant_pnl
        self._total_reward += reward

        # Game over logic
        try:
            self._prices_history.append(next(self._data_generator))
        except StopIteration:
            done = True
            info['status'] = 'No more data.'
        if self._iteration >= self._episode_length:
            done = True
            info['status'] = 'Time out.'
            
        return self.get_state(), reward, done, info
    
    def _handle_close(self, evt):
        self._closed_plot = True
            
    def is_valid(self, action, position):
        actions_allowed = _allowed[position]
        allowed = False
        for x in actions_allowed:
            if _actions[x] == action:
                allowed = True
        return allowed
            
    def get_order_value(self):
        ask = self.current["asks"][0]
        bid = self.current["bids"][0]
        return (ask + bid)/2
            
    def send_order_to_buy(self):
        self._position = self._positions['ordened_buy']
        self._entry_price = self.get_order_value()
            
    def send_order_to_sell(self):
        self._position = self._positions['ordened_sell']
        self._exit_price = self.get_order_value()

    def cancel_buy(self):
        self._position = self._positions['flat']

    def cancel_sell(self):
        self._position = self._positions['flat']

    def get_observation(self):
        self.current = self._data_generator(index)
        current_price = self.current["price"]
        #Checking for passive position changes
        if self._position == self._positions['ordened_sell'] and current_price >= self._exit_price:
            reward -= self._trading_fee
            self._position = self._positions['flat']
            self.instant_pnl = self._exit_price - self._entry_price
            self._entry_price = 0
        elif all(self._position == self._positions['ordened_buy']) and current_price <= self._entry_price:
            reward -= self._trading_fee
            self._position = self._positions['bought']

    def get_output_state(self):
        return self.current

            