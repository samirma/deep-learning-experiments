import numpy as np
import datetime

_actions = {
    'hold': 0,
    'buy': 1,
    'sell': 2,
    'cancel_buy': 3,
    'cancel_sell': 4
}

_positions = {
    'flat': 0,
    'ordened_buy': 1,
    'ordened_sell': 2,
    'bought': 3
}

_allowed = {
    _positions['flat']: {'hold', 'buy'},
    _positions['bought']: {'hold', 'sell'},
    _positions['ordened_sell']: {'hold', 'cancel_sell'},
    _positions['ordened_buy']: {'hold', 'cancel_buy'}
}

# integer encode input data
def onehot_encoded (integer_encoded, char_to_int=_positions):
    # one hot encode
    onehot_encoded = list()
    letter = [0 for _ in range(len(char_to_int))]
    letter[integer_encoded] = 1
    onehot_encoded.append(letter)
    
    return onehot_encoded[0]


class TraderEnv():
    """Class for a discrete (buy/hold/sell) spread trading environment.
    """

    def __init__(self, data_generator, episode_length=1000, trading_fee=0, time_fee=0, history_length=2):
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
        self.instant_pnl = 0
        self._position = _positions['flat']
        self._entry_price = 0
        self._exit_price = 0

        observation = self.get_observation()
        self._action = _actions['hold']
        return observation

    def step_string(self, action_string):
        return self.step(_actions[action_string])
    
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
        self.reward = -self._time_fee
        
        observation = self.get_observation()
        
        if self.is_valid(action, self._position):
            if action == _actions['buy']:
                self.send_order_to_buy()
            elif action == _actions['sell']:
                self.send_order_to_sell()
            elif action == _actions['cancel_buy']:
                self.cancel_buy()
            elif action == _actions['cancel_sell']:
                self.cancel_sell()
            
        else:
            info['status'] = 'Invalid action'
            done = True
            self.reward = -1000
            
        self.reward += self.instant_pnl
        self._total_pnl += self.instant_pnl
        self._total_reward += self.reward

        # Game over logic
        if self._data_generator.has_next() == False:
            done = True
            info['status'] = 'No more data.'
        if self._iteration >= self._episode_length:
            done = True
            info['status'] = 'Time out.'
            
        return self.get_state(), self.reward, done, info
    
    def _handle_close(self, evt):
        self._closed_plot = True
            
    def get_order_value(self):
        ask = float(self.current["asks"][0][0])
        bid = float(self.current["bids"][0][0])
        result = (ask + bid)/2
        return result
            
    def send_order_to_buy(self):
        print("send_order_to_buy")
        self._position = _positions['ordened_buy']
        self._entry_price = self.get_order_value()
        print(self._entry_price)
            
    def send_order_to_sell(self):
        self._position = _positions['ordened_sell']
        self._exit_price = self.get_order_value()

    def cancel_buy(self):
        self._position = _positions['flat']

    def cancel_sell(self):
        self._position = _positions['flat']

    def get_current_state(self):
        return self.current

    def get_current_position(self):
        return self._position
    
    def get_state(self):
        raw_state = self.current
        list = []
        bids = raw_state["bids"]
        asks = raw_state["asks"]

        price = raw_state["price"]

        #list.append(price)
        list.append(raw_state["amount"])

        def prepare_orders(orders, multi):
            for order in orders:
                list.append((float(order[0])/price) * multi)
                list.append(float(order[1]))

        prepare_orders(asks, 1)
        prepare_orders(bids, -1)
        
        list.extend(onehot_encoded(self.get_current_position()))
        
        return list
        
    def get_observation(self):
        self.current = self._data_generator.next()
        current_price = self.current["price"]
        
        #value = datetime.datetime.fromtimestamp(int(self.current["timestamp"]))
        #print(value.strftime('%Y-%m-%d %H:%M:%S'))
        
        #Checking for passive position changes
        if self._position == _positions['ordened_sell'] and current_price >= self._exit_price:
            self.reward -= self._trading_fee
            self._position = _positions['flat']
            self.instant_pnl = self._exit_price - self._entry_price
            self._entry_price = 0
            print(self.instant_pnl)
        elif self._position == _positions['ordened_buy'] and current_price <= self._entry_price:
            self.reward -= self._trading_fee
            self._position = _positions['bought']

    def get_output_state(self):
        return self.current

    def is_valid(self, action, position):
        actions_allowed = _allowed[position]
        allowed = False
        for x in actions_allowed:
            if _actions[x] == action:
                allowed = True
        return allowed
            