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

class Net (nn.Module):
    def __init__(self, obs_space, actions):
        super().__init__()

        self.fc1 = nn.Linear(obs_space, 128) #fully connected layter #1
        self.fc2 = nn.Linear(128, 128) #fully connected layer 2
        self.output = nn.Linear(128, actions) #hideen -> output

    def forward(self, x):
        x = F.relu(self.fc1(x)) #activation
        x = F.relu(self.fc2(x)) #activation
        return self.output(x) #output

class ExperienceMemory():
    def __init__(self,maxlength=1000):
        self.memory = deque([], maxlength)

    def append(self, next_experience):
        self.memory.append(next_experience)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)
    
    