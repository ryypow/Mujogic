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
        self.action_space = spaces.Discrete(5)
        
        # Movement magnitudes
        self.strong = 0.5
        self.weak = 0.15
        
        # Joint indices by finger
        # [.1 knuckle, .2 spread, .3 mid-curl, .4 tip-curl]
        self.finger1 = [0, 1, 2, 3]
        self.finger2 = [4, 5, 6, 7]    # THE PUSHER
        self.finger3 = [8, 9, 10, 11]
        self.thumb = [12, 13, 14, 15]
        
        # Holders = everyone except finger2
        self.holders = self.finger1 + self.finger3 + self.thumb

    def action(self, act):
        continuous = np.zeros(16)

        if act == 0:  # GRASP - tighten all holders
            for finger in [self.finger1, self.finger3, self.thumb]:
                continuous[finger[2]] = self.strong   # .3 mid-curl
                continuous[finger[3]] = self.strong   # .4 tip-curl

        elif act == 1:  # RELEASE - loosen holders
            for finger in [self.finger1, self.finger3, self.thumb]:
                continuous[finger[2]] = -self.strong
                continuous[finger[3]] = -self.strong

        elif act == 2:  # ROTATE +Z (counter-clockwise from above)
            continuous[self.finger2[0]] = self.strong
            continuous[self.finger2[1]] = -self.strong  # was positive, flip to negative
            continuous[self.finger2[2]] = self.strong
            continuous[self.finger2[3]] = self.weak
            
            for finger in [self.finger1, self.finger3, self.thumb]:
                continuous[finger[2]] = -self.weak

        elif act == 3:  # HOLD - maintain current position
            continuous[:] = 0.0

        elif act == 4:  # ROTATE -Z (clockwise from above)
            continuous[self.finger2[0]] = self.strong
            continuous[self.finger2[1]] = self.strong   # was negative, flip to positive
            continuous[self.finger2[2]] = self.strong
            continuous[self.finger2[3]] = self.weak
            
            for finger in [self.finger1, self.finger3, self.thumb]:
                continuous[finger[2]] = -self.weak

        return continuous
