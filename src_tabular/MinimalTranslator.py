"""
MinimalTranslator.py - Sequential rotation action space

Strategy (from visual testing):
  - Thumb (joint 12) pushes first
  - Thumb (joint 12) retracts to get out of the way
  - Finger 3 curls tip (joints 9, 10, 11) then nudges with joint 8
  - Finger 2 (joint 4) comes down to push further

Finger groups in the 16-joint array:
  [0:4]   = Finger 1 - STATIC
  [4:8]   = Finger 2 - joint 4 is PUSHER
  [8:12]  = Finger 3 - curl tip (9,10,11), nudge with 8
  [12:16] = Thumb - joint 12 is PUSHER

Actions:
  0: HOLD            - Everything static
  1: THUMB_PUSH      - Thumb pushes in (joint 12 +)
  2: THUMB_RETRACT   - Thumb pulls back (joint 12 -)
  3: FINGER3_CURL    - Finger 3 curls tip (joints 9, 10, 11 +)
  4: FINGER3_NUDGE   - Finger 3 nudges with joint 8
  5: FINGER3_RETRACT - Finger 3 retracts all joints
  6: FINGER2_PUSH    - Finger 2 comes down (joint 4 +)
  7: FINGER2_RETRACT - Finger 2 pulls back (joint 4 -)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MinimalTranslator(gym.ActionWrapper):
    """
    Sequential rotation with thumb, finger3, and finger2.

    Joints that move:
    - Joint 12 (Thumb MCP) - push and retract
    - Joints 8, 9, 10, 11 (Finger 3) - curl tip then nudge
    - Joint 4 (Finger 2 first joint) - push further
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(8)  # 8 actions now

        # The joints that move
        self.FINGER2_J4 = 4    # Finger 2 first joint
        self.FINGER3_J8 = 8    # Finger 3 first joint (nudge)
        self.FINGER3_J9 = 9    # Finger 3 second joint (curl)
        self.FINGER3_J10 = 10  # Finger 3 third joint (curl)
        self.FINGER3_J11 = 11  # Finger 3 fourth joint (curl)
        self.THUMB_MCP = 12    # Thumb base - pushes and retracts

        # Movement magnitudes
        self.push_delta = 0.10
        self.curl_delta = 0.15  # Stronger curl for tip
        self.nudge_delta = 0.12  # Sharp nudge

    def action(self, action):
        """
        Convert discrete action to 16-joint continuous array.
        """
        continuous = np.zeros(16)

        if action == 0:  # HOLD - Everything static
            pass

        elif action == 1:  # THUMB_PUSH - Thumb pushes in
            continuous[self.THUMB_MCP] = self.push_delta

        elif action == 2:  # THUMB_RETRACT - Thumb pulls back
            continuous[self.THUMB_MCP] = -self.push_delta

        elif action == 3:  # FINGER3_CURL - Curl the tip (joints 9, 10, 11)
            continuous[self.FINGER3_J9] = self.curl_delta
            continuous[self.FINGER3_J10] = self.curl_delta
            continuous[self.FINGER3_J11] = self.curl_delta

        elif action == 4:  # FINGER3_NUDGE - Sharp nudge with joint 8
            continuous[self.FINGER3_J8] = self.nudge_delta

        elif action == 5:  # FINGER3_RETRACT - Pull back all joints
            continuous[self.FINGER3_J8] = -self.nudge_delta
            continuous[self.FINGER3_J9] = -self.curl_delta
            continuous[self.FINGER3_J10] = -self.curl_delta
            continuous[self.FINGER3_J11] = -self.curl_delta

        elif action == 6:  # FINGER2_PUSH - Finger 2 comes down
            continuous[self.FINGER2_J4] = self.push_delta

        elif action == 7:  # FINGER2_RETRACT - Finger 2 pulls back
            continuous[self.FINGER2_J4] = -self.push_delta

        return continuous


class MinimalTranslator3Action(gym.ActionWrapper):
    """
    Simplified 3-action version (no separate retract).

    Actions:
      0: HOLD          - Everything static
      1: THUMB_PUSH    - Thumb pushes
      2: FINGER3_SWIPE - Finger 3 swipes (and thumb retracts)
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(3)

        self.FINGER3_J10 = 10
        self.THUMB_MCP = 12

        self.push_delta = 0.10
        self.swipe_delta = 0.10

    def action(self, action):
        continuous = np.zeros(16)

        if action == 0:  # HOLD
            pass

        elif action == 1:  # THUMB_PUSH
            continuous[self.THUMB_MCP] = self.push_delta

        elif action == 2:  # FINGER3_SWIPE (thumb retracts at same time)
            continuous[self.THUMB_MCP] = -self.push_delta
            continuous[self.FINGER3_J10] = self.swipe_delta

        return continuous
