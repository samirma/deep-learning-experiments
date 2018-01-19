#https://github.com/Grzego/async-rl/tree/master/a3c

from scipy.misc import imresize
from skimage.color import rgb2gray
from multiprocessing import *
from collections import deque
import gym
import numpy as np
import h5py
import traceback
import sys
from numpy import argmax
from pathlib import Path
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.models import Sequential

game = "trade"
model_file = 'save_model/model-trade.h5'

def build_network(input_shape, output_shape):
    from keras.models import Model
    from keras.layers import Input, Conv2D, Flatten, Dense, Dropout
    # -----
    dropout_rate = 0.25

    state = Input(shape=input_shape)

    window_size = 10

    h = Bidirectional(LSTM(window_size, return_sequences=True), input_shape=(window_size, state.shape[-1]),)(h)
    h = Dropout(dropout_rate)(h)

    #Second recurrent layer with dropout
    h = Bidirectional(LSTM((window_size*2), return_sequences=True))(h)
    h = Dropout(dropout_rate)(h)

    #Third recurrent layer
    h = Bidirectional(LSTM(window_size, return_sequences=False))(h)

    h = Dense(128, activation='relu')(h)
    h = Dropout(dropout_rate)(h)
    h = Flatten()(h)
    h = Dense(128*3, activation='relu')(h)
    h = Dropout(dropout_rate)(h)
    h = Dense(128, activation='relu')(h)
    h = Dropout(dropout_rate)(h)
    h = Dense(128, activation='relu')(h)
    h = Dropout(dropout_rate)(h)
    h = Dense(64, activation='relu')(h)
    h = Dropout(dropout_rate)(h)

    value = Dense(1, activation='linear', name='value')(h)
    policy = Dense(output_shape, activation='softmax', name='policy')(h)

    value_network = Model(inputs=state, outputs=value)
    policy_network = Model(inputs=state, outputs=policy)

    adventage = Input(shape=(1,))
    train_network = Model(inputs=[state, adventage], outputs=[value, policy])

    return value_network, policy_network, train_network, adventage


def policy_loss(adventage=0., beta=0.01):
    from keras import backend as K

    def loss(y_true, y_pred):
        return -K.sum(K.log(K.sum(y_true * y_pred, axis=-1) + K.epsilon()) * K.flatten(adventage)) + \
               beta * K.sum(y_pred * K.log(y_pred + K.epsilon()))

    return loss


def value_loss():
    from keras import backend as K

    def loss(y_true, y_pred):
        return 0.5 * K.sum(K.square(y_true - y_pred))

    return loss


# -----

class LearningAgent(object):
    def __init__(self, action_space, observation_shape, batch_size=32, swap_freq=200):
        from keras.optimizers import RMSprop		
        # -----
        
        self.input_depth = 1
        self.past_range = 20
        self.observation_shape = (self.input_depth * self.past_range,) + observation_shape
        self.batch_size = batch_size
        self.beta = 0.01

        _, _, self.train_net, adventage = build_network(self.observation_shape, action_space.n)

        self.train_net.compile(optimizer='adam',
                               loss=[value_loss(), policy_loss(adventage, self.beta)])

        self.pol_loss = deque(maxlen=25)
        self.val_loss = deque(maxlen=25)
        self.values = deque(maxlen=25)
        self.entropy = deque(maxlen=25)
        self.swap_freq = swap_freq
        self.swap_counter = self.swap_freq
        self.unroll = np.arange(self.batch_size)
        self.targets = np.zeros((self.batch_size, action_space.n))
        self.counter = 0

    def learn(self, last_observations, actions, rewards, learning_rate=0.001):
        import keras.backend as K
        K.set_value(self.train_net.optimizer.lr, learning_rate)
        frames = len(last_observations)
        self.counter += frames
        # -----
        values, policy = self.train_net.predict([last_observations, self.unroll])
        # -----
        self.targets.fill(0.)
        adventage = rewards - values.flatten()
        self.targets[self.unroll, actions] = 1.
        # -----
        loss = 0
        loss = self.train_net.train_on_batch([last_observations, adventage], [rewards, self.targets])
        entropy = np.mean(-policy * np.log(policy + 0.00000001))
        self.pol_loss.append(loss[2])
        self.val_loss.append(loss[1])
        self.entropy.append(entropy)
        self.values.append(np.mean(values))
        if False:
            min_val, max_val, avg_val = min(self.values), max(self.values), np.mean(self.values)
            print('\rFrames: %8d; Policy-Loss: %10.6f; Avg: %10.6f '
                    '--- Value-Loss: %10.6f; Avg: %10.6f '
                    '--- Entropy: %7.6f; Avg: %7.6f '
                    '--- V-value; Min: %6.3f; Max: %6.3f; Avg: %6.3f' % (
                        self.counter,
                        loss[2], np.mean(self.pol_loss),
                        loss[1], np.mean(self.val_loss),
                        entropy, np.mean(self.entropy),
                        min_val, max_val, avg_val), end='')
        # -----
        self.swap_counter -= frames
        if self.swap_counter < 0:
            self.swap_counter += self.swap_freq
            return True
        return False
    
    def save_model(self):
        print("Saving the model")
        self.train_net.save_weights(model_file, overwrite=True)
        print("Model saved")

def learn_proc(mem_queue, weight_dict, get_enviroment):
    import os
    pid = os.getpid()
    print(' %5d> Learning process' % (pid,))
    os.environ['THEANO_FLAGS'] = 'floatX=float32,device=gpu,nvcc.fastmath=False,lib.cnmem=0.3,' + \
                                 'compiledir=th_comp_learn'

    # -----
    save_freq = 10000
    learning_rate = 0.0001
    batch_size = 32
    checkpoint = 0
    steps = 99000000
    # -----
    env = get_enviroment()
    state_size = env.observation_space.shape
    agent = LearningAgent(env.action_space, state_size, batch_size=batch_size)
    try:         
        # -----
        if Path(model_file).exists():
            print(' %5d> Loading weights from file' % (pid,))
            agent.train_net.load_weights(model_file)
            # -----
        print(' %5d> Setting weights in dict' % (pid,))
        weight_dict['update'] = 0
        weight_dict['weights'] = agent.train_net.get_weights()
        # -----
        last_obs = np.zeros((batch_size,) + agent.observation_shape)
        actions = np.zeros(batch_size, dtype=np.int32)
        rewards = np.zeros(batch_size)
        # -----
        idx = 0
        agent.counter = checkpoint
        save_counter = checkpoint % save_freq + save_freq
        while True:
            
            last_obs[idx, ...], actions[idx], rewards[idx] = mem_queue.get()
            idx = (idx + 1) % batch_size
            if idx == 0:
                lr = max(0.00000001, (steps - agent.counter) / steps * learning_rate)
                updated = agent.learn(last_obs, actions, rewards, learning_rate=lr)
                if updated:
                    # print(' %5d> Updating weights in dict' % (pid,))
                    weight_dict['weights'] = agent.train_net.get_weights()
                    weight_dict['update'] += 1
            # -----
            save_counter -= 1
            if save_counter < 0:
                save_counter += save_freq
                agent.save_model()
    except Exception:
        print ('learn_proc Exception')
        exc_type, exc_value, exc_tb = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_tb)

class ActingAgent(object):
    def __init__(self, env, n_step=8, discount=0.99):
        
        #print("ActingAgent", env)

        action_space = env.action_space
        
        self.input_depth = 1
        self.past_range = 20
        self.observation_shape = (self.input_depth * self.past_range,) + env.observation_space.shape

        #print("pre build_network" , action_space, self.observation_shape)

        self.value_net, self.policy_net, self.load_net, adv = build_network(self.observation_shape, action_space.n)

        #print("pos build_network")

        self.value_net.compile(optimizer='adam', loss='mse')
        self.policy_net.compile(optimizer='adam', loss='categorical_crossentropy')
        self.load_net.compile(optimizer='adam', loss='mse', loss_weights=[0.5, 1.])  # dummy loss

        self.action_space = action_space
        self.observations = np.zeros(self.observation_shape)
        self.last_observations = np.zeros_like(self.observations)
        # -----
        self.n_step_observations = deque(maxlen=n_step)
        self.n_step_actions = deque(maxlen=n_step)
        self.n_step_rewards = deque(maxlen=n_step)
        self.n_step = n_step
        self.discount = discount
        self.counter = 0

    def init_episode(self, observation):
        for _ in range(self.past_range):
            self.save_observation(observation)

    def reset(self):
        self.counter = 0
        self.n_step_observations.clear()
        self.n_step_actions.clear()
        self.n_step_rewards.clear()

    def sars_data(self, action, reward, observation, terminal, mem_queue):
        self.save_observation(observation)
        reward = np.clip(reward, -1., 1.)
        # reward /= reward_scale
        # -----
        self.n_step_observations.appendleft(self.last_observations)
        self.n_step_actions.appendleft(action)
        self.n_step_rewards.appendleft(reward)
        # -----
        self.counter += 1
        if terminal or self.counter >= self.n_step:
            r = 0.
            if not terminal:
                r = self.value_net.predict(self.observations[None, ...])[0]
            for i in range(self.counter):
                r = self.n_step_rewards[i] + self.discount * r
                mem_queue.put((self.n_step_observations[i], self.n_step_actions[i], r))
            self.reset()

    def choose_action(self):
        policy = self.policy_net.predict(self.observations[None, ...])[0]
        action = np.random.choice(np.arange(self.action_space.n), p=policy)
        return action

    def choose_action_from_observation(self, observation):
        self.save_observation(observation)
        policy = self.policy_net.predict(self.observations[None, ...])[0]
        policy /= np.sum(policy)  # numpy, why?
        return argmax(policy)
    
    def save_observation(self, observation):
        self.last_observations = self.observations[...]
        self.observations = np.roll(self.observations, -self.input_depth, axis=0)
        self.observations[-self.input_depth:, ...] = observation



def generate_experience_proc(mem_queue, weight_dict, no, generator):
    import os
    pid = os.getpid()
    os.environ['THEANO_FLAGS'] = 'floatX=float32,device=gpu,nvcc.fastmath=True,lib.cnmem=0,' + \
                                 'compiledir=th_comp_act_' + str(no)
    # -----
    print(' %5d> Process started' % (pid,))
    # -----
    frames = 0
    batch_size = 32
    # -----
    
    env = generator()

    agent = ActingAgent(env, n_step=5)

    if frames > 0:
        agent.load_net.load_weights(model_file)
        print(' %5d> Loaded weights from file' % (pid,))
    else:
        import time
        while 'weights' not in weight_dict:
            time.sleep(0.1)
        agent.load_net.set_weights(weight_dict['weights'])
        print(' %5d> Loaded weights from dict' % (pid,))

    
    best_score = 0
    avg_score = deque([0], maxlen=25)

    last_update = 0
    try:

        while True:
            #print("while")
            done = False
            episode_reward = 0
            op_last, op_count = 0, 0
            observation = env.reset()
            #print("init_episode")
            agent.init_episode(observation)
            
            # -----
            while not done:
                frames += 1
                action = agent.choose_action()
                observation, reward, done, info = env.step(action)
                #print(info)
                episode_reward += env.total_profite
                best_score = max(best_score, episode_reward)
                # -----
                agent.sars_data(action, reward, observation, done, mem_queue)
                # -----
                #op_count = 0 if op_last != action else op_count + 1
                #done = done or op_count >= 100
                op_last = action
                # -----
                if frames % 2000 == 0:
                    print('%5d> Best: %6.3f; Avg: %6.3f; Max: %6.3f' % (
                        pid, best_score, np.mean(avg_score), np.max(avg_score)))
                if frames % batch_size == 0:
                    update = weight_dict.get('update', 0)
                    if update > last_update:
                        last_update = update
                        # print(' %5d> Getting weights from dict' % (pid,))
                        agent.load_net.set_weights(weight_dict['weights'])
            # -----
            avg_score.append(episode_reward)
    except Exception:
        print ('generate_experience_proc execption')
        exc_type, exc_value, exc_tb = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_tb)

def init_worker():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main():
    manager = Manager()
    weight_dict = manager.dict()
    mem_queue = manager.Queue(queue_size)

    pool = Pool(processes + 1, init_worker)

    try:
        for i in range(processes):
            pool.apply_async(generate_experience_proc, (mem_queue, weight_dict, i))

        pool.apply_async(learn_proc, (mem_queue, weight_dict))

        pool.close()
        pool.join()

    except:
        pool.terminate()
        pool.join()
        raise


