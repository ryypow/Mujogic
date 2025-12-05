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
        self.action_space = spaces.Discrete(3) #two action features - GRASP and ROTATE

    def action(self, act):
        """
        Convert discrete action to continuous actions
        two discrete actions:
            - action 0 = grasp, which closes all fingers
            - action 1 = rotate, which makes half of the joints close and half open
            - action 2 = hold, keeps jpoints in current position
        """
        continuous = np.zeros(16)

        if act == 0: #grasp
            continuous[:] = -0.02

        elif act == 1: #rotate - half open and half close
            continuous[0:4] = -0.02 #finger 1 and 2 extends
            continuous[4:8] = 0.01 #finger 2 closes slightly
            continuous[8:12] = 0.02 #finger 3  closes 
            continuous[12:16] = 0.00 #thumb anchors

        elif act == 2: #hold at target
            continuous[:] = 0.0 #this will keep the fingers in their current position, stopping rotation
            #NOTE: this may not work due to the momentum of the cubes rotation

        return continuous
    
    
