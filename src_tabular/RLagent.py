# RLagent.py - Q-Learning Agent for Cube Rotation
import numpy as np
import os
import mujoco
from scipy.spatial.transform import Rotation

class QLearningAgent:
    """
    Tabular Q-Learning agent for discrete state/action spaces.
    State: 18 states (3 speed bins x 6 progress bins)
    Actions: 5 discrete actions (GRASP, RELEASE, ROT_POS, HOLD, ROT_NEG)
    """

    def __init__(self, num_states=18, num_actions=5, learning_rate=0.1,
                 discount=0.99, epsilon=0.8, epsilon_decay=0.997, min_epsilon=0.05):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Initialize Q-table with zeros
        self.q_table = np.zeros((num_states, num_actions))

        self.action_names = {
            0: "HOLD",
            1: "THUMB_PUSH",
            2: "THUMB_RETRACT",
            3: "FINGER3_CURL",
            4: "FINGER3_NUDGE",
            5: "FINGER3_RETRACT",
            6: "FINGER2_PUSH",
            7: "FINGER2_RETRACT"
        }

    def get_state(self, env):
        """
        Convert environment observation to discrete state ID.
        State = speed_bin * 6 + progress_bin (18 total states)
        """
        base_env = env.unwrapped

        # Get Z rotation
        z_rot = base_env.get_object_z_rotation()  # Returns degrees

        # Goal progress (how far from 90 degrees)
        GOAL = 90
        goal_progress = abs(GOAL - z_rot)
        progress_ratio = goal_progress / GOAL

        # 6 progress bins
        if progress_ratio > 0.80:
            progress_bin = 0  # very far
        elif progress_ratio > 0.60:
            progress_bin = 1  # far
        elif progress_ratio > 0.40:
            progress_bin = 2  # medium
        elif progress_ratio > 0.20:
            progress_bin = 3  # close
        elif progress_ratio > 0.10:
            progress_bin = 4  # very close
        else:
            progress_bin = 5  # at goal

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

        # State ID: 18 states (3 speed x 6 progress)
        state_id = speed_bin * 6 + progress_bin

        return state_id, z_rot, goal_progress

    def get_action(self, state):
        """
        Epsilon-greedy action selection.
        """
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.num_actions)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def learn(self, state, action, reward, next_state, done):
        """
        Q-learning update: Q(s,a) = Q(s,a) + lr * (r + gamma * max(Q(s')) - Q(s,a))
        """
        if done:
            best_future_q = 0
        else:
            best_future_q = np.max(self.q_table[next_state])

        current_q = self.q_table[state, action]
        new_q = current_q + self.learning_rate * (reward + self.discount * best_future_q - current_q)
        self.q_table[state, action] = new_q

    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Save Q-table to file."""
        np.save(filepath, self.q_table)
        print(f"Q-table saved to {filepath}")

    def load(self, filepath):
        """Load Q-table from file."""
        if os.path.exists(filepath):
            loaded = np.load(filepath)
            if loaded.shape == self.q_table.shape:
                self.q_table = loaded
                print(f"Q-table loaded from {filepath}")
                return True
            else:
                print(f"Shape mismatch! Expected {self.q_table.shape}, got {loaded.shape}")
        else:
            print(f"No Q-table found at {filepath}")
        return False
