import os
import json
import io


class DataGenerator:
    def __init__(self):
        self.base_dir = "stock_data/"
        self.files = os.listdir(self.base_dir)
        self.files.sort()
        self.steps = len(self.files)
        self.index = 0

    def next(self):
        file_path = self.base_dir + self.files[self.index]
        f = io.open(file_path, mode="r", encoding="utf-8")
        raw = f.read()
        self.index += 1
        return json.loads(raw)
    
    def rewind(self):
        self.index = 0
        
    def has_next(self):
        return self.index < self.steps
    
    def max_steps(self):
        return self.steps