import numpy as np
import datetime
from gym import error, spaces
from numpy import argmax

_actions = {
    'hold': 0,
    'buy': 1,
    'sell': 2
    #'cancel_buy': 3,
    #'cancel_sell': 4
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
    #_positions['ordened_sell']: {'hold', 'cancel_sell'},
    #_positions['ordened_buy']: {'hold', 'cancel_buy'}
    _positions['ordened_sell']: {'hold', 'sell'},
    _positions['ordened_buy']: {'hold', 'buy'}
}

# integer encode input data
def onehot_encoded (integer_encoded, char_to_int=_positions):
    # one hot encode
    onehot_encoded = list()
    letter = [0 for _ in range(len(char_to_int))]
    letter[integer_encoded] = 1
    onehot_encoded.append(letter)
    
    return onehot_encoded[0]


def print_log(string):
    log = string
    #print(log)


class TraderEnv():
    """Class for a discrete (buy/hold/sell) spread trading environment.
    """

    def __init__(self, data_generator, episode_length=6000, trading_fee=0, time_fee=0, history_length=3, stage_history_length=3):
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
        
        #assert history_length > 0
        self._data_generator = data_generator
        self._first_render = True
        self.total_profite = 0
        self._iteration = 0
        self._trading_fee = trading_fee
        self._time_fee = time_fee
        self._episode_length = data_generator.max_steps()
        self.action_space = spaces.Discrete(len(_actions))
        self._prices_history = []
        self._positions_history = []
        self._history_length = history_length
        self.stage_history_length = stage_history_length
        self.reset()

    def reset(self):
        """Reset the trading environment. Reset rewards, data generator...
        Returns:
            observation (numpy.array): observation of the state
        """
        
        self._prices_history = []
        self._positions_history = []
        
        self._data_generator.rewind()
        self._total_reward = 0
        self._position = _positions['flat']
        self._entry_price = 0
        self._exit_price = 0
        
        self.total_profite = 0
        self._iteration = 0
        
        self.invalid_actions = 0
        self.max_invalid_actions = 100
        
        self.minimum_reward_limit = -500
        self.invalid_actions_punishiment = -300
                
        self.get_observation()
        
        for i in range(self.stage_history_length): 
            self._positions_history.append(self._position)
            self.get_observation()
            
        self.observation_space = spaces.Box(-1,1,len(self.get_state()))
        
        self._action = _actions['hold']
        return self.get_state()

    def step_string(self, action_string):
        return self.step(_actions[action_string])
    
    def step(self, encoded_action):
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
        
        #print_log(encoded_action)
        #action = argmax(encoded_action)
        action = encoded_action
        
        #print(action)
        
        self._action = action
        self._iteration += 1
        self.done = False
        self.info = {}
        self.reward = 0
        
        observation = self.get_observation()
        
        if not self.done: 
            if self.is_valid(action, self._position):
                if action == _actions['buy']:
                    self.send_order_to_buy()
                elif action == _actions['sell']:
                    self.send_order_to_sell()
                #elif action == _actions['cancel_buy']:
                #    self.cancel_buy()
                #elif action == _actions['cancel_sell']:
                #    self.cancel_sell()
                #elif action == _actions['hold']:
                #    self.add_reward(-0.0001)
            else:
                self.info['status'] = 'Invalid action'
                self.invalid_actions += 1
                if self.max_invalid_actions < self.invalid_actions:
                    self.done = True
                    self.add_reward(self.invalid_actions_punishiment)
                    print_log("max_invalid_actions: %s invalid_actions: %s" % (self.max_invalid_actions, self.invalid_actions))
                else:
                    self.add_reward(-0.02)
        
        if not self.done:
            self.verify_position()
        
        self._total_reward += self.reward

        # Game over logic
        if not self._data_generator.has_next():
            self.done = True
            self.info['status'] = 'No more data.'
        elif self._total_reward < self.minimum_reward_limit:
            self.done = True
            self.info['status'] = 'reward too low %s' % (self._total_reward)
        
        self.add_reward(-self._time_fee)
        
        state, reward, done, info = self.get_state(), self.reward, self.done, self.info
        
        if self.done:
            self.game_over()
        
        self.info['iteration'] = self._iteration
        self.info['profite'] = self.total_profite
        self.info['invalid_actions'] = self.invalid_actions
        
        return state, reward, done, info 
    
    def game_over(self):
        print_log("Game over(%s): Profite(%s) Reward(%s) Total reward(%s) info: %s" % (self._iteration, self.total_profite, self.reward, self._total_reward, self.info))
        #self.reset()
        
    
    def _handle_close(self, evt):
        self._closed_plot = True
            
    def get_order_sell_value(self):
        bid = float(self.current["bids"][0][0])
        return bid
            
    def get_order_buy_value(self):
        ask = float(self.current["asks"][0][0])
        return ask

    def send_order_to_buy(self):
        if self._position == _positions['flat']:
            self.add_reward(0.01)
            self._position = _positions['ordened_buy']
            self._entry_price = self.get_order_buy_value()
            #print("send_order_to_buy: Current price %s order: %s" % (profite, self.reward))
            
    def send_order_to_sell(self):
        if self._position == _positions['bought']:
            self.add_reward(0.01)
            self._position = _positions['ordened_sell']
            self._exit_price = self.get_order_sell_value()
            #print("send_order_to_sell: Current price %s order: %s" % (profite, self.reward))

    def cancel_buy(self):
        self.add_reward(1)
        self._position = _positions['flat']

    def cancel_sell(self):
        self.add_reward(1)
        self._position = _positions['flat']

    def get_current_state(self):
        return self.current

    def get_current_position(self):
        return self._position
    
    def get_state(self):
        raw_state = self.current
        list = []

        price = raw_state["price"]

        def prepare_orders(orders, price, multi):
            for order in orders:
                list.append((float(order[0])/price) * multi)
                #list.append(float(order[1]))

        self._prices_history = self._prices_history[-self.stage_history_length:]
        for old_order in self._prices_history:
            bids = old_order["bids"][:5]
            asks = old_order["asks"][:5]
            prepare_orders(asks, price, 1)
            prepare_orders(bids, price, -1)
            
        self._positions_history.append(self._position)
        self._positions_history = self._positions_history[-self.stage_history_length:]
        for old_positions in self._positions_history:
            list.extend(onehot_encoded(old_positions))
        
        if self._entry_price != 0:
            list.extend([price/self._entry_price])
        else:
            list.extend([0])
        
        return np.array(list)
        
    def get_observation(self):
        self.current = self._data_generator.next()
        self._prices_history.append(self.current)
        current_price = self.current["price"]
                
        if self._position == _positions['ordened_sell']:
            #self.add_reward(0.001)
            if current_price >= self._exit_price:
                self.info['status'] = 'Order sold'
                self._position = _positions['flat']
                profite = self._exit_price - self._entry_price
                self.total_profite += profite
                self._entry_price = 0
                self._exit_price = 0
                has_profite = profite > 0
                self.done = not has_profite
                if has_profite:
                    reward = 10 + profite*10
                    self.add_reward(reward)
                    #print("Profite: %s current reward %s, total reward %s" % (profite, self.reward, self._total_reward))
                else:
                    no_profite = -10 + profite*10
                    self.add_reward(no_profite)

        elif self._position == _positions['ordened_buy']:
            #self.reward -= self._trading_fee
            if current_price <= self._entry_price:
                #self.add_reward(0.001)
                self._position = _positions['bought']

    def get_output_state(self):
        return self.current
    
    def render(self, mode = "test"):
        return True

    def verify_position(self):
        price = self.current["price"]
        
        prices = []
        
        prices.append(price)

        for old_order in self._prices_history:
            old_price = old_order["price"]
            prices.append(old_price)
        
        average = (sum(prices) / len(prices))
        
        price_is_getting_higher = price > average
        
        if (price_is_getting_higher):
            if self._position == _positions['flat'] or self._position == _positions['ordened_sell']:
                penault = abs((price / average) - 1)
                #print("Getting higher: Current price(%s) > average(%s) = (%s) " % (price, average, penault))
                self.add_reward(-penault)
            else:
                self.add_reward(0.00001)
        else: #price is getting lower
            if self._position == _positions['bought'] or self._position == _positions['ordened_buy']:
                penault = abs((price / average) - 1)
                #print("Getting lower: Current price(%s) < average(%s) = (%s) " % (price, average, penault))
                self.add_reward(-penault)
            else:
                self.add_reward(0.00001)
    
    def add_reward(self, reward):
        #print("add: ", reward)
        self.reward += reward
    
    def is_valid(self, action, position):
        actions_allowed = _allowed[position]
        allowed = False
        for x in actions_allowed:
            if _actions[x] == action:
                allowed = True
        return allowed
            