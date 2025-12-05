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
import mujoco

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

class ObsTranslator(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
#TODO       self.observation_space = spaces.Discrete(54)

    def observation(self, observation):
        state_id, _, _, _, _ = self._get_discrete(observation)
        return state_id
    
    def _get_discrete(self, observation):
        base_env = self.env.unwrapped
    
        #get z-rot
        z_rot = base_env.get_object_z_rotation()
        z_rot_rad = np.deg2rad(z_rot)

        # Goal progress (how far from target)
        GOAL = 90
        goal_rad = np.deg2rad(90)
        goal_progress = abs(GOAL - z_rot)

        # RELATIVE progress bins - 6 bins for finer control
        progress_ratio = goal_progress / GOAL

        # 6 progress bins (was 4) - more granular for better learning
        if progress_ratio > 0.80:
            progress_bin = 0  # very far (>80% remaining)
        elif progress_ratio > 0.60:
            progress_bin = 1  # far (60-80% remaining)
        elif progress_ratio > 0.40:
            progress_bin = 2  # medium (40-60% remaining)
        elif progress_ratio > 0.20:
            progress_bin = 3  # close (20-40% remaining)
        elif progress_ratio > 0.10:
            progress_bin = 4  # very close (10-20% remaining)
        else:
            progress_bin = 5  # at goal (<10% remaining)

        # Speed bin (angular velocity)
        obj_vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            base_env.sim.model, base_env.sim.data,
            mujoco.mjtObj.mjOBJ_BODY, base_env.obj_body_id, obj_vel, 0
        )
        angular_velocity_z = abs(obj_vel[2])

        if angular_velocity_z < 0.02:
            speed_bin = 0  # still
        elif angular_velocity_z < 0.1:
            speed_bin = 1  # rotating
        else:
            speed_bin = 2  # fast

        # State ID: 18 states
#TODO        state_id = speed_bin * 6 + progress_bin

        return state_id, progress_bin, speed_bin, z_rot, goal_progress
