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
        self.action_space = spaces.Discrete(5)
        self.strong_movement = 0.05
        self.weak_movement = 0.025

    def action(self, act):
        continuous = np.zeros(16)

        if act == 0: #strong grasp - close all
            continuous[:] = self.strong_movement
        
        elif act == 1: #strong release - open all
            continuous[:] = -self.strong_movement

        elif act == 2: #rotation POSITIVE (+Z direction)
            continuous[0:4] = -self.strong_movement   # stronger push to move cube more
            continuous[4:8] = -self.weak_movement  #slightly releases
            continuous[8:12] = self.strong_movement  * 1.5  # finger3 OPENS (allows rotation)
            continuous[12:16] = self.weak_movement    # thumb stabilizes

        elif act == 3: #hold at target
            continuous[:] = 0.0 #this will keep the fingers in their current position, stopping rotation
            #NOTE: this may not work due to the momentum of the cubes rotation

        #elif act == 4: #rotation NEGATIVE (-Z direction)
        #    continuous[0:4] = -self.strong_movement   # finger1 OPENS (allows rotation)
        #    continuous[4:8] = 0.00                    # finger2 anchored
        #    continuous[8:12] = self.strong_movement   # finger3 CLOSES (pushes cube -Z)
        #    continuous[12:16] = self.weak_movement    # thumb stabilizes

        return continuous
    
    
