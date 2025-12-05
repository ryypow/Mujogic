import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ActionTranslator(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        # Discrete macro actions
        self.action_space = spaces.Discrete(5)

        # This controls how aggressively we move toward target poses
        # (tune between 0.05 and 0.2; 0.1 is a good starting point)
        self.speed = 0.1  
        self.debug_counts = {a: 0 for a in range(5)}  # 0..4


        # Cache base env action limits to clip into
        base_env = self.env.unwrapped
        self.base_low = base_env.action_space.low
        self.base_high = base_env.action_space.high

        # Starting position (professor's tripod grasp)
        self.start_pose = np.array([
            1.0, 0.3, 1.0, 1.0,   # finger1
            0.0, 0.0, 0.0, 0.0,   # finger2 (relaxed)
            1.3, 1.0, 1.3, 1.0,   # finger3
            0.8, 1.3, 0.8, 0.5    # thumb
        ], dtype=np.float32)

        # Slightly tighter grasp
        self.grasp_pose = np.array([
            1.0, 0.3, 1.2, 1.2,
            0.3, 0.0, 0.3, 0.3,
            1.3, 1.0, 1.5, 1.2,
            0.8, 1.3, 1.0, 0.7
        ], dtype=np.float32)

        # Finger2 positions
        self.finger2_ready = np.array([0.3, 0.0, 0.3, 0.3], dtype=np.float32)
        self.finger2_push_pos = np.array([1.2, -0.6, 1.2, 1.0], dtype=np.float32)
        self.finger2_push_neg = np.array([1.2,  0.6, 1.2, 1.0], dtype=np.float32)
        # Holder positions (fingers 1, 3, thumb)
        
        self.holders_tight = np.array([
            1.3, 0.3, 1.3, 1.2,   # finger1 anchor
            1.1, 1.0, 1.0, 0.8,   # finger3
            0.7, 1.1, 0.6, 0.4    # thumb
        ], dtype=np.float32)

        self.holders_loose = np.array([
            1.0, 0.3, 0.8, 0.8,   # finger1
            1.3, 1.0, 1.0, 0.8,   # finger3
            0.8, 1.3, 0.6, 0.4    # thumb
        ], dtype=np.float32)

    def action(self, act: int):
        base_env = self.env.unwrapped

        # Current hand joint angles (16-D)
        current = np.array([
            base_env.sim.data.qpos[base_env.sim.model.jnt_qposadr[j]]
            for j in base_env.sim.hand_joint_ids
        ], dtype=np.float32)

        target = current.copy()

        if act == 0:  # GRASP (tighten around object)
            target = self.grasp_pose.copy()

        elif act == 1:  # RELEASE (back to starting tripod pose)
            target = self.start_pose.copy()

        elif act == 2:  # ROTATE +Z
            target[4:8] = self.finger2_push_pos

            # brace / pivot
            target[0:4] = self.holders_tight[0:4]
            target[8:12] = self.holders_loose[4:8]
            target[12:16] = self.holders_loose[8:12]

        elif act == 3:  # HOLD (no change)
            delta = np.zeros_like(current, dtype=np.float32)
            # Still clip to be safe
            delta = np.clip(delta, self.base_low, self.base_high).astype(np.float32)
            return delta

        elif act == 4:  # ROTATE -Z
            target[4:8] = self.finger2_push_neg

            target[0:4] = self.holders_tight[0:4]
            target[8:12] = self.holders_loose[4:8]
            target[12:16] = self.holders_loose[8:12]

        # Move a fraction toward the target pose
        delta = (target - current) * self.speed

        # CRITICAL: respect the base env’s action_space
        delta = np.clip(delta, self.base_low, self.base_high).astype(np.float32)


        delta = (target - current) * self.speed

        # Clip to base env range if you’re doing that
        base_env = self.env.unwrapped
        delta = np.clip(delta,
                        base_env.action_space.low,
                        base_env.action_space.high).astype(np.float32)
        
        if self.debug_counts[act] < 20:  # up to 20 prints per action type
            print(
                f"[ActionTranslator] act={act}, "
                f"delta_norm={np.linalg.norm(delta):.4f}, "
                f"delta_min={delta.min():.4f}, delta_max={delta.max():.4f}"
            )
            self.debug_counts[act] += 1
        # after computing and clipping delta
        if self.debug_counts[act] < 20:  # 20 prints per action type
            print(
                f"[ActionTranslator] act={act}, "
                f"delta_norm={np.linalg.norm(delta):.4f}, "
                f"delta_min={delta.min():.4f}, delta_max={delta.max():.4f}"
            )
            self.debug_counts[act] += 1

        return delta


