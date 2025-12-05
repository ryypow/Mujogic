import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ActionTranslator(gym.ActionWrapper):
    """
    Very simple delta-based discrete action wrapper.

    - act=0: GRASP (close all fingers a bit)
    - act=1: RELEASE (open all fingers a bit)
    - act=2: ROT_POS (push with "finger2" while tightening other fingers)
    - act=3: HOLD (no change)
    - act=4: ROT_NEG (reverse push)
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(5)

        base_env = self.env.unwrapped
        self.base_low = base_env.action_space.low
        self.base_high = base_env.action_space.high

        # Magnitudes (all within [-0.03, 0.03])
        self.grasp_step   = 0.02
        self.release_step = -0.02
        self.rot_push     = 0.03
        self.rot_brace    = 0.01

        # Debug counters
        self.debug_counts = {a: 0 for a in range(5)}

    def action(self, act: int):
        # 16 hand joints
        delta = np.zeros(16, dtype=np.float32)

        # Convention: indices
        # 0..3  = finger 1
        # 4..7  = finger 2 (pusher)
        # 8..11 = finger 3
        # 12..15 = thumb
        # (this matches your 16-length poses & grasp angles)

        if act == 0:  # GRASP: close everything a bit
            delta[:] = self.grasp_step

        elif act == 1:  # RELEASE: open everything a bit
            delta[:] = self.release_step

        elif act == 2:  # ROT_POS: push with finger2, brace with others
            # Push with finger2 joints
            # Adjust signs for your geometry; this is a good starting guess:
            delta[4] += self.rot_push    # curl
            delta[5] -= self.rot_push    # spread / yaw inward
            delta[6] += self.rot_push    # curl
            delta[7] += self.rot_push    # extra flex

            # Light tightening on other fingers as a pivot
            delta[0:4]  += self.rot_brace    # finger1
            delta[8:12] += self.rot_brace    # finger3
            delta[12:16] += self.rot_brace   # thumb

        elif act == 3:  # HOLD
            delta[:] = 0.0

        elif act == 4:  # ROT_NEG: opposite push
            delta[4] -= self.rot_push
            delta[5] += self.rot_push
            delta[6] -= self.rot_push
            delta[7] -= self.rot_push

            delta[0:4]  += self.rot_brace
            delta[8:12] += self.rot_brace
            delta[12:16] += self.rot_brace

        # Clip to base env action space
        base_env = self.env.unwrapped
        delta = np.clip(delta,
                        base_env.action_space.low,
                        base_env.action_space.high).astype(np.float32)

        # Debug (up to 10 prints per action type)
        if self.debug_counts[act] < 10:
            print(
                f"[ActionTranslator] act={act}, "
                f"delta_norm={np.linalg.norm(delta):.4f}, "
                f"delta_min={delta.min():.4f}, "
                f"delta_max={delta.max():.4f}"
            )
            self.debug_counts[act] += 1

        return delta
