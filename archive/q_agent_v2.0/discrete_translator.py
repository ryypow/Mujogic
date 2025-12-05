"""
This translator will convert 5 discrete actions into the 16 individual movements for all 16 joints
based on the OpenAI Gym wrapper tutorial

Actions:
  0: GRASP - close all fingers
  1: RELEASE - open all fingers
  2: ROT_POS - rotate cube in +Z direction
  3: HOLD - maintain current position
  4: ROT_NEG - rotate cube in -Z direction
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ActionTranslator(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Define the discrete action space
        self.action_space = spaces.Discrete(5)
        # INCREASED: Stronger movements for more effective rotation
        self.strong_movement = 0.08  # Was 0.05
        self.weak_movement = 0.04    # Was 0.025
        self.rotation_boost = 2.0    # Extra multiplier for rotation actions

    def action(self, act):
        continuous = np.zeros(16)

        if act == 0:  # GRASP - close all fingers
            continuous[:] = self.strong_movement

        elif act == 1:  # RELEASE - open all fingers
            continuous[:] = -self.strong_movement

        elif act == 2:  # ROT_POS - rotate in +Z direction
            # Finger groups: [0:4]=finger1, [4:8]=finger2, [8:12]=finger3, [12:16]=thumb
            # Key insight: need asymmetric push/pull to create torque
            continuous[0:4] = -self.strong_movement * self.rotation_boost  # finger1 OPENS strongly
            continuous[4:8] = self.weak_movement                           # finger2 holds light contact
            continuous[8:12] = self.strong_movement * self.rotation_boost  # finger3 PUSHES strongly (+Z)
            continuous[12:16] = self.strong_movement * 0.5                 # thumb anchors

        elif act == 3:  # HOLD - maintain position
            continuous[:] = 0.0

        elif act == 4:  # ROT_NEG - rotate in -Z direction (mirror of ROT_POS)
            continuous[0:4] = self.strong_movement * self.rotation_boost   # finger1 PUSHES strongly (-Z)
            continuous[4:8] = self.weak_movement                           # finger2 holds light contact
            continuous[8:12] = -self.strong_movement * self.rotation_boost # finger3 OPENS strongly
            continuous[12:16] = self.strong_movement * 0.5                 # thumb anchors

        return continuous
