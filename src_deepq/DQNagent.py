import numpy as np
import os
import mujoco
from scipy.spatial.transform import Rotation as R
import matplotlib as plt
import random
import torch
from torch import nn
import torch.nn.functional as F
from collections import deque

class DQNmodel (nn.Module):
    def __init__(self, obs, hidden, actions):
        super().__init__()

        self.fullyconnected1 = nn.Linear(obs, hidden) #fully connected layter #1
        self.output = nn.Linear(hidden, actions)

    def forward(self, x):
        x = F.relu(self.fullyconnected1(x)) #activation
        x = self.out(x) #output

class ExperienceMemory():
    def __init__(self):
        self.memory = deque([], maxlength=maxlength)

    def append(self, next_experience):
        self.memory.append(next_experience)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)
    
    