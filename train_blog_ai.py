
import os
import json
import io
from data_generator import DataGenerator
from trade_env import TraderEnv
from blog_ai import A3CAgent
from async_rl import *
import time
import numpy as np
from multiprocessing import *
from collections import deque

import random

stage_length_var = 10
def get_enviroment():
        chance = (random.randint(1,100))

        if chance > 50:
            return TraderEnv(DataGenerator(random=False, first_index=10000), stage_history_length=stage_length_var)
        else:
            return TraderEnv(DataGenerator(random=True, first_index=10000), stage_history_length=stage_length_var)

env = get_enviroment()

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

global_agent = A3CAgent(state_size, action_size, "TraderEnv", get_enviroment, threads=16)

global_agent.train()
