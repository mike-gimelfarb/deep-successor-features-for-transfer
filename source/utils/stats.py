# -*- coding: UTF-8 -*-
import numpy as np


class OnlineMeanVariance:
    
    def __init__(self):
        self.count = 0
        self.mean = 0.
        self.M2 = 0.
    
    def update(self, x):
        x = np.array(x)
        self.count += 1
        delta = x - self.mean
        self.mean = self.mean + delta / float(self.count)
        delta2 = x - self.mean
        self.M2 = self.M2 + delta * delta2
    
    def calculate_variance(self):
        return self.M2 / (self.count - 1.)
    
    def calculate_standard_error(self):
        return np.sqrt(self.calculate_variance() / float(self.count))

