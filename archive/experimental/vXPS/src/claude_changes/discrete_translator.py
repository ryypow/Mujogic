"""
Discrete action translator for in-hand cube rotation.
Converts discrete action IDs to continuous joint movements.

Action Space (6 actions):
    0: GRASP      - Close all fingers (tighten grip)
    1: RELEASE    - Open all fingers slightly (loosen grip)
    2: ROTATE_CW  - Rotate clockwise (fingers 1&2 open, 3&thumb close)
    3: ROTATE_CCW - Rotate counter-clockwise (opposite of CW)
    4: FINE_CW    - Fine clockwise adjustment (smaller movement)
    5: HOLD       - Maintain current position
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ActionTranslator(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        # Action parameters
        self.strong = 0.025  # Strong movement delta
        self.fine = 0.01     # Fine movement delta

        # 6 discrete actions
        self.action_space = spaces.Discrete(6)

        # Action names for debugging
        self.action_names = {
            0: "GRASP",
            1: "RELEASE",
            2: "ROTATE_CW",
            3: "ROTATE_CCW",
            4: "FINE_CW",
            5: "HOLD"
        }

    def action(self, act):
        """
        Convert discrete action to 16-dimensional continuous joint deltas.

        Joint layout:
            [0:4]   - Finger 1 (index finger)
            [4:8]   - Finger 2 (middle finger)
            [8:12]  - Finger 3 (ring finger)
            [12:16] - Thumb

        NEGATIVE values = close/curl finger (towards palm)
        POSITIVE values = open/extend finger (away from palm)
        """
        continuous = np.zeros(16, dtype=np.float32)

        # NOTE: NEGATIVE = close/curl, POSITIVE = open/extend

        if act == 0:  # GRASP - close all fingers
            continuous[:] = -self.strong

        elif act == 1:  # RELEASE - open all fingers
            continuous[:] = self.strong

        elif act == 2:  # ROTATE_CW - clockwise rotation
            # Based on working config: finger1 extends, finger2 closes slightly,
            # finger3 closes, thumb anchors
            continuous[0:4] = self.strong      # Finger 1 extends (pushes)
            continuous[4:8] = -self.fine       # Finger 2 closes slightly
            continuous[8:12] = -self.strong    # Finger 3 closes (pushes)
            continuous[12:16] = 0.0            # Thumb anchors

        elif act == 3:  # ROTATE_CCW - counter-clockwise rotation
            # Opposite
            continuous[0:4] = -self.strong     # Finger 1 closes
            continuous[4:8] = self.fine        # Finger 2 opens slightly
            continuous[8:12] = self.strong     # Finger 3 extends
            continuous[12:16] = 0.0            # Thumb anchors

        elif act == 4:  # FINE_CW - fine clockwise adjustment
            continuous[0:4] = self.fine
            continuous[8:12] = -self.fine

        elif act == 5:  # HOLD - no movement
            continuous[:] = 0.0

        return continuous

    def get_action_name(self, act):
        """Return human-readable action name."""
        return self.action_names.get(act, "UNKNOWN")
