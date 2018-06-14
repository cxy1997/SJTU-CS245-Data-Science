import numpy as np
import torch
class people():
    def __init__(self, age):
        self.age = torch.zeros(1, 8)
        self.age[0, age // 10 - 1] = 1

p = people(89)
print(p.age)