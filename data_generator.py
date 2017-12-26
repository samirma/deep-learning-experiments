import os
import json
import io
import random

class DataGenerator:
    def __init__(self, random=True, first_index=0):
        self.base_dir = "stock_data/"
        self.files = os.listdir(self.base_dir)
        self.files.sort()
        self.steps = len(self.files)
        self.first_index = first_index
        self.index = self.first_index
        self.is_random = random
        self.index = first_index

    def next(self):
        #print("next %s %s %s" % (self.index, self.steps, self.has_next()))
        file_path = self.base_dir + self.files[self.index]
        f = io.open(file_path, mode="r", encoding="utf-8")
        raw = f.read()
        self.index += 1
        return json.loads(raw)
    
    def rewind(self):
        if self.is_random:
            self.first_index = int(random.uniform(0, 0.9)*self.steps)
        self.index = self.first_index
        print("Initial index %s, has_next: %s" % (self.index, self.has_next()))
        
    def has_next(self):
        has_next = ((self.index) < self.steps)
        #print("has_next %s" % has_next)
        return has_next
    
    def max_steps(self):
        return self.steps