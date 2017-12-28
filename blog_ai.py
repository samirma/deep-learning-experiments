import threading
import numpy as np
import tensorflow as tf
import pylab
import time
import gym
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K


# global variables for threading
episode = 0
scores = []

EPISODES = 324906

# This is A3C(Asynchronous Advantage Actor Critic) agent(global) for the Cartpole
# In this example, we use A3C algorithm
class A3CAgent:
    def __init__(self, state_size, action_size, env_name, env_generator):
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        
        self.get_enviroment = env_generator
                
        episode = 0
        scores = []

        # get gym environment name
        self.env_name = env_name

        # these are hyper parameters for the A3C
        self.actor_lr = 0.00001
        self.critic_lr = 0.00001
        self.discount_factor = 0.7

        self.threads = 16

        # create model for actor and critic network
        self.actor, self.critic = self.build_model()

        # method for training actor and critic network
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

    # approximate policy and value using Neural Network
    # actor -> state is input and probability of each action is output of network
    # critic -> state is input and value of state is output of network
    # actor and critic network share first hidden layer
    def build_model(self):
        state = Input(batch_shape=(None,  self.state_size))
        shared = Dense(self.state_size, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform')(state)
        
        def dense(outputs, activation, inputs):
            hidden = Dense(outputs, activation=activation, kernel_initializer='glorot_uniform')(inputs)
            return hidden
        
        actor_hidden = dense(32, 'relu', shared)
        actor_hidden = dense(32, 'relu', actor_hidden)
        actor_hidden = dense(32, 'relu', actor_hidden)
        actor_hidden = dense(32, 'relu', actor_hidden)
        actor_hidden = dense(32, 'relu', actor_hidden)
        actor_hidden = dense(16, 'relu', actor_hidden)
        actor_hidden = dense(16, 'relu', actor_hidden)
        action_prob = Dense(self.action_size, activation='softmax', kernel_initializer='glorot_uniform')(actor_hidden)

        value_hidden = dense(32, 'relu', shared)
        value_hidden = dense(32, 'relu', value_hidden)
        value_hidden = dense(32, 'relu', value_hidden)
        value_hidden = dense(32, 'relu', value_hidden)
        value_hidden = dense(16, 'relu', value_hidden)
        value_hidden = dense(16, 'relu', value_hidden)
        value_hidden = dense(16, 'relu', value_hidden)
        state_value = Dense(1, activation='linear', kernel_initializer='he_uniform')(value_hidden)

        actor = Model(inputs=state, outputs=action_prob)
        critic = Model(inputs=state, outputs=state_value)

        actor._make_predict_function()
        critic._make_predict_function()

        #actor.summary()
        #critic.summary()

        return actor, critic

    # make loss function for Policy Gradient
    # [log(action probability) * advantages] will be input for the back prop
    # we add entropy of action probability to loss
    def actor_optimizer(self):
        action = K.placeholder(shape=(None, self.action_size))
        advantages = K.placeholder(shape=(None, ))

        policy = self.actor.output

        good_prob = K.sum(action * policy, axis=1)
        eligibility = K.log(good_prob + 1e-10) * K.stop_gradient(advantages)
        loss = -K.sum(eligibility)

        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)

        actor_loss = loss + 0.01*entropy

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], actor_loss)
        train = K.function([self.actor.input, action, advantages], [], updates=updates)
        return train

    # make loss function for Value approximation
    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(None, ))

        value = self.critic.output

        loss = K.mean(K.square(discounted_reward - value))

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, discounted_reward], [], updates=updates)
        return train

    # make agents(local) and start training
    def train(self):
        # self.load_model('./save_model/cartpole_a3c.h5')
        agents = [Agent(i, self.actor, self.critic, self.optimizer, self.env_name, self.discount_factor,
                        self.action_size, self.state_size, self.get_enviroment) for i in range(self.threads)]

        for agent in agents:
            agent.start()

        for agent in agents:
            agent.join()

        plot = scores[:]
        pylab.plot(range(len(plot)), plot, 'b')
        pylab.savefig("./save_graph/model.png")

        self.save_model('./save_model/model')
        
        
    def save_model(self, name):
        self.actor.save_weights(name + "_actor.h5")
        self.critic.save_weights(name + "_critic.h5")

    def load_model(self, name):
        self.actor.load_weights(name + "_actor.h5")
        self.critic.load_weights(name + "_critic.h5")

# This is Agent(local) class for threading
class Agent(threading.Thread):
    def __init__(self, index, actor, critic, optimizer, env_name, discount_factor, action_size, state_size, get_enviroment):
        threading.Thread.__init__(self)

        self.states = []
        self.rewards = []
        self.actions = []

        self.index = index
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.env_name = env_name
        self.discount_factor = discount_factor
        self.action_size = action_size
        self.state_size = state_size
        self.get_enviroment = get_enviroment

    # Thread interactive with environment
    def run(self):
        global episode
        env = self.get_enviroment()
        while episode < EPISODES:
            state = env.reset()
            score = 0
            
            while True:
                action = self.get_action(state)
                next_state, reward, done, info = env.step(action)
                score += reward

                self.memory(state, action, reward)

                state = next_state

                if done or score < env.minimum_reward_limit:
                    episode += 1
                    print("episode: ", episode, "/ score : ", score, "| end : ", info)
                    scores.append(score)
                    self.train_episode(True)
                    break
        
    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def discount_rewards(self, rewards, done=True):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        if not done:
            running_add = self.critic.predict(np.reshape(self.states[-1], (1, self.state_size)))[0]
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save <s, a ,r> of each step
    # this is used for calculating discounted rewards
    def memory(self, state, action, reward):
        self.states.append(state)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)

    # update policy network and value network every episode
    def train_episode(self, done):
        discounted_rewards = self.discount_rewards(self.rewards, done)

        values = self.critic.predict(np.array(self.states))
        values = np.reshape(values, len(values))

        advantages = discounted_rewards - values

        self.optimizer[0]([self.states, self.actions, advantages])
        self.optimizer[1]([self.states, discounted_rewards])
        self.states, self.actions, self.rewards = [], [], []

    def get_action(self, state):
        policy = self.actor.predict(np.reshape(state, [1, self.state_size]))[0]
        return np.random.choice(self.action_size, 1, p=policy)[0]
