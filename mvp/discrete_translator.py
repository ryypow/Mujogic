"""
This translator will convert 2 discrete actions into the 16 individual movements for all 16 joints
based on the OpenAI Gym wrapper tutorial
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ActionTranslator(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        #define the discrete action spaces
        self.action_space = spaces.Discrete(2) #two action features - GRASP and ROTATE

    def action(self, act):
        """
        Convert discrete action to continuous actions
        two discrete actions:
            - action 0 = grasp, which closes all fingers
            - action 1 = rotate, which makes half of the joints close and half open
        """
        continuous = np.zeros(16)

        if act == 0: #grasp
            continuous[:] = -0.03

        elif act == 1: #rotate - half open and half close
            continuous[0:8] = 0.03 #finger 1 and finger 2 joints
            continuous[8:16] = -0.03 #finger 3 and thumb joints

        return continuous
    
    
