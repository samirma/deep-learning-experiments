
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



stage_length_var = 30
def get_enviroment():
    return TraderEnv(DataGenerator(), stage_history_length=stage_length_var)

manager = Manager()
weight_dict = manager.dict()
mem_queue = manager.Queue(256)

threads = 16

try:

    pool = Pool(threads + 1, init_worker)

    for i in range(threads):
        pool.apply_async(generate_experience_proc, (mem_queue, weight_dict, i, get_enviroment))

    pool.apply_async(learn_proc, (mem_queue, weight_dict, get_enviroment))

    pool.close()
    pool.join()

except:
    pool.terminate()
    pool.join()
    raise

