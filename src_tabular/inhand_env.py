# inhand_env.py (v3
import os
import numpy as np
import mujoco
import mujoco.viewer as mjv
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation as R

from simulation import Simulation

MAX_EPISODE_STEPS = 300
TARGET_ROTATION = -90.0  # degrees (negative direction based on testing)
TARGET_TOLERANCE = 5.0   # degrees - how close is "success"


class CanRotateEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, render_mode=None):
        super(CanRotateEnv, self).__init__()
        
        # Initialize simulation and get object IDs
        self.sim = Simulation(
            scene_path=os.path.join(os.path.dirname(__file__), "scene.xml"),
            output_dir="rl_output"
        )
        self.sim.load()
        self.obj_body_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, "obj1") #
        self.site_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site") #
        self.sim.ids_by_name(["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"], mujoco.mjtObj.mjOBJ_JOINT, 'arm') #
        self.sim.ids_by_name(["1", "0", "2", "3", "5", "4", "6", "7", "9", "8", "10", "11", "12", "13", "14", "15"], mujoco.mjtObj.mjOBJ_JOINT, 'hand') #
        self.sim.actuators_for_joints('arm') #
        self.sim.actuators_for_joints('hand') #
        self.can_geom_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_GEOM, "obj1")
        self.fingertip_geom_ids = {
            mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_GEOM, "fingertip"),
            mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_2"),
            mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_GEOM, "thumb_fingertip"),
        }
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(5)
        self.action_space = spaces.Discrete(5)
        obs_size = len(self.sim.hand_joint_ids) + 7
# TODO        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
# TODO        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        self.render_mode = render_mode
        if render_mode == 'human':
            self.viewer = mjv.launch_passive(self.sim.model, self.sim.data)
            self.viewer = mjv.launch_passive(self.sim.model, self.sim.data)
        else:
            self.viewer = None
        self.step_count = 0
        self.rotation_goal_delta = TARGET_ROTATION  # Total rotation needed
        self.prev_z_rotation = None  # Track previous rotation for reward calc

    def get_object_z_rotation(self):
        """
        Calculates the Z-axis rotation of an object from its quaternion.
        """
        # Get the quaternion (w, x, y, z) from MuJoCo data
        quat_wxyz = self.sim.data.xquat[self.obj_body_id]
        
        # Scipy's Rotation object expects (x, y, z, w)
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        
        # Create a Rotation object
        r = R.from_quat(quat_xyzw)
        
        # Convert to Euler angles (xyz order) in degrees
        euler_angles_deg = r.as_euler('xyz', degrees=True)
        
        # Return the Z-axis rotation
        return euler_angles_deg[2]

    def print_object_status(self):
        """Prints the object's current position and Z rotation."""
        
        # We must call mj_forward() to ensure all physics-derived
        # values (like xpos and xquat) are up-to-date.
        mujoco.mj_forward(self.sim.model, self.sim.data)
                
        # Get object position
        obj_pos = self.sim.data.xpos[self.obj_body_id]
        
        # Get object Z rotation
        obj_z_rot = self.get_object_z_rotation()
        
        # Print to console
        print(f"  > Object Position (x, y, z):  ({obj_pos[0]:.4f}, {obj_pos[1]:.4f}, {obj_pos[2]:.4f})")
        print(f"  > Object Z Rotation (degrees): {obj_z_rot:.2f}°")
        self.rotation_goal_delta = TARGET_ROTATION  # Total rotation needed
        self.prev_z_rotation = None  # Track previous rotation for reward calc

    def get_object_z_rotation(self):
        """
        Calculates the Z-axis rotation of an object from its quaternion.
        """
        # Get the quaternion (w, x, y, z) from MuJoCo data
        quat_wxyz = self.sim.data.xquat[self.obj_body_id]
        
        # Scipy's Rotation object expects (x, y, z, w)
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        
        # Create a Rotation object
        r = R.from_quat(quat_xyzw)
        
        # Convert to Euler angles (xyz order) in degrees
        euler_angles_deg = r.as_euler('xyz', degrees=True)
        
        # Return the Z-axis rotation
        return euler_angles_deg[2]

    def print_object_status(self):
        """Prints the object's current position and Z rotation."""
        
        # We must call mj_forward() to ensure all physics-derived
        # values (like xpos and xquat) are up-to-date.
        mujoco.mj_forward(self.sim.model, self.sim.data)
                
        # Get object position
        obj_pos = self.sim.data.xpos[self.obj_body_id]
        
        # Get object Z rotation
        obj_z_rot = self.get_object_z_rotation()
        
        # Print to console
        print(f"  > Object Position (x, y, z):  ({obj_pos[0]:.4f}, {obj_pos[1]:.4f}, {obj_pos[2]:.4f})")
        print(f"  > Object Z Rotation (degrees): {obj_z_rot:.2f}°")

    def _get_obs(self):
        finger_qpos = np.array([self.sim.data.qpos[self.sim.model.jnt_qposadr[j]] for j in self.sim.hand_joint_ids]) #
        obj_jnt_adr = self.sim.model.body_jntadr[self.obj_body_id] #
        obj_qpos_adr = self.sim.model.jnt_qposadr[obj_jnt_adr] #
        object_pose = self.sim.data.qpos[obj_qpos_adr : obj_qpos_adr + 7] #
        return np.concatenate([finger_qpos, object_pose])

    def _calculate_reward(self, cube_rotation_new, cube_rotation_prev):


        obj_vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.sim.model, self.sim.data, mujoco.mjtObj.mjOBJ_BODY, self.obj_body_id, obj_vel, 0)
        angular_velocity_z = obj_vel[2]

        current_distance = abs(TARGET_ROTATION - cube_rotation_new)
        goal_distance = abs(self.rotation_goal_delta)  # Total rotation needed (e.g., 45°, 60°, 90°)

        # === Percentage-based progress with exponential scaling ===
        progress_pct = 1.0 - (current_distance / goal_distance) if goal_distance > 0 else 1.0
        progress_pct = max(0.0, min(1.0, progress_pct))  # Clamp to [0, 1]

        # Exponential scaling: reward increases significantly as we get closer
        # At 50% progress: multiplier ~1.5, at 90%: multiplier ~2.6
        progress_multiplier = 1.0 + (progress_pct ** 2) * 2.0

        if cube_rotation_prev is not None:
            previous_distance = abs(TARGET_ROTATION - cube_rotation_prev)
            progress_reward = (previous_distance - current_distance) * 100.0 * progress_multiplier
        else:
            progress_reward = 0.0

        # === Milestone bonuses (curriculum-friendly) ===
        milestone_bonus = 0.0
        milestones = [0.25, 0.50, 0.75, 0.90]  # 25%, 50%, 75%, 90% of goal
        milestone_values = [15.0, 35.0, 60.0, 100.0]  # Increasing rewards

        if cube_rotation_prev is not None:
            prev_pct = 1.0 - (abs(TARGET_ROTATION - cube_rotation_prev) / goal_distance) if goal_distance > 0 else 1.0
            prev_pct = max(0.0, min(1.0, prev_pct))
            for threshold, bonus in zip(milestones, milestone_values):
                # Award bonus when crossing a milestone threshold
                if prev_pct < threshold <= progress_pct:
                    milestone_bonus += bonus

        # === Scaled completion bonus ===
        if current_distance < TARGET_TOLERANCE:
            # Base 150 + up to 150 more for hitting it precisely
            precision_bonus = (1.0 - current_distance / TARGET_TOLERANCE) * 150.0
            nearTarget_bonus = 150.0 + precision_bonus
        else:
            nearTarget_bonus = 0.0

        # Direction-aware rotation reward
        direction_to_target = np.sign(TARGET_ROTATION - cube_rotation_new)

        if cube_rotation_prev is not None:
            rotation_change = cube_rotation_new - cube_rotation_prev
            # Increased base rotation reward
            rotation_reward = direction_to_target * rotation_change * 80.0

            # Wrong direction penalty (stronger)
            moving_wrong_way = (direction_to_target > 0 and rotation_change < 0) or \
                              (direction_to_target < 0 and rotation_change > 0)
            drift_penalty = abs(rotation_change) * 70.0 if moving_wrong_way else 0.0
        else:
            rotation_reward = 0.0
            drift_penalty = 0.0

        # ===Stagnation penalty ===
        stagnation_penalty = 0.0
        if cube_rotation_prev is not None:
            rotation_change_abs = abs(cube_rotation_new - cube_rotation_prev)
            # If barely moving AND far from goal, penalize
            if rotation_change_abs < np.deg2rad(0.3) and progress_pct < 0.80:
                # Stronger penalty when further from goal
                stagnation_penalty = 1.0 * (1.0 - progress_pct)

        # Survival reward
        can_pos = self.sim.data.xpos[self.obj_body_id]
        palm_pos = self.sim.data.site_xpos[self.site_id]
        distance_from_palm = np.linalg.norm(can_pos - palm_pos)
        survival_reward = 0.01 - (distance_from_palm * 0.5)  # Reduced
        survival_reward = 0.01 - (distance_from_palm * 0.5)  # Reduced

        # Contact reward
        # Contact reward
        contact_reward = 0.0
        fingers_in_contact = set()
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2
            if geom1 in self.fingertip_geom_ids and geom2 == self.can_geom_id:
                fingers_in_contact.add(geom1)
            elif geom2 in self.fingertip_geom_ids and geom1 == self.can_geom_id:
                fingers_in_contact.add(geom2)


        if len(fingers_in_contact) >= 3:
            contact_reward = 0.3
            contact_reward = 0.3
        elif len(fingers_in_contact) > 0:
            contact_reward = 0.08 * len(fingers_in_contact)

        # === Total reward calculation ===
        total_reward = (progress_reward + milestone_bonus + nearTarget_bonus +
                       rotation_reward + survival_reward + contact_reward
                        - drift_penalty - stagnation_penalty)

        contact_reward = 0.08 * len(fingers_in_contact)

        # === Total reward calculation ===
        total_reward = (progress_reward + milestone_bonus + nearTarget_bonus +
                       rotation_reward + survival_reward + contact_reward
                        - drift_penalty - stagnation_penalty)

        return total_reward

    def _is_terminated(self):
        can_z_pos = self.sim.data.xpos[self.obj_body_id][2] #
        palm_z_pos = self.sim.data.site_xpos[self.site_id][2] #
        return can_z_pos < (palm_z_pos - 0.05) or self.step_count > MAX_EPISODE_STEPS

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.prev_z_rotation = None  # Reset rotation tracking for new episode

        mujoco.mj_resetData(self.sim.model, self.sim.data)

        target_pos_up = np.array([0.4, 0.0, .5]) #
        target_euler_up = np.array([0, 0, 0]) #
        q_palm_up = self.sim.desired_qpos_from_ik(self.site_id, target_pos_up, target_euler_up) #
        self.sim.set_joint_positions(self.sim.arm_joint_ids, q_palm_up) #
        for i, act_id in enumerate(self.sim.arm_act_ids):
            self.sim.data.ctrl[act_id] = q_palm_up[i] #
        
        mujoco.mj_forward(self.sim.model, self.sim.data)

        palm_surface_pos = self.sim.data.site_xpos[self.site_id].copy() #
        object_start_pos = palm_surface_pos + np.array([0.011, -0.03, 0.075]) #
        obj_jnt_adr = self.sim.model.body_jntadr[self.obj_body_id] #
        obj_qpos_adr = self.sim.model.jnt_qposadr[obj_jnt_adr] #
        self.sim.data.qpos[obj_qpos_adr : obj_qpos_adr + 3] = object_start_pos #
        self.sim.data.qpos[obj_qpos_adr + 3 : obj_qpos_adr + 7] = [1, 0, 0, 0] #

        mujoco.mj_forward(self.sim.model, self.sim.data)

        for _ in range(20):
            mujoco.mj_step(self.sim.model, self.sim.data) #

        q_open_angles = np.array(  [1.0, -0.3, 0.7, 0.7,  # Finger 1 (open)
            1.0, -0.3, 0.8, 0.8,   # Finger 2
            1.3, 1.5, 1.0, 1.4,   # Finger 3 
            -0.3, 1.4, 1.0, 2.0]) #thumb
        
        q_open_angles = np.array(  [1.0, -0.3, 0.7, 0.7,  # Finger 1 (open)
            1.0, -0.3, 0.8, 0.8,   # Finger 2
            1.3, 1.5, 1.0, 1.4,   # Finger 3 
            -0.3, 1.4, 1.0, 2.0]) #thumb
        
        self.sim.set_joint_positions(self.sim.hand_joint_ids, q_open_angles) #
        for i, act_id in enumerate(self.sim.hand_act_ids):
            self.sim.data.ctrl[act_id] = q_open_angles[i] #

        mujoco.mj_forward(self.sim.model, self.sim.data)

        if self.render_mode != "headless":
            self.viewer.sync()
        
        return self._get_obs(), {}

    def step(self, action):
        # Get rotation BEFORE action
        z_rot_before = self.get_object_z_rotation()

        # Get rotation BEFORE action
        z_rot_before = self.get_object_z_rotation()

        target_angles = np.array([self.sim.data.qpos[self.sim.model.jnt_qposadr[j]] for j in self.sim.hand_joint_ids]) + action
        self.sim.move_gripper_to_angles(target_angles, 0.5)
        self.sim.move_gripper_to_angles(target_angles, 0.5)

        if self.render_mode != "headless":
            self.viewer.sync()

        self.step_count += 1

        # Get rotation AFTER action
        z_rot_after = self.get_object_z_rotation()


        # Get rotation AFTER action
        z_rot_after = self.get_object_z_rotation()

        observation = self._get_obs()
        reward = self._calculate_reward(z_rot_after, z_rot_before)
        reward = self._calculate_reward(z_rot_after, z_rot_before)
        terminated = self._is_terminated()
        truncated = self.step_count >= MAX_EPISODE_STEPS

        # Store for next step
        self.prev_z_rotation = z_rot_after

        # Store for next step
        self.prev_z_rotation = z_rot_after

        return observation, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch(self.sim.model, self.sim.data)
            
            # Check if the viewer is still active before trying to sync
            try:
                if self.viewer.is_running():
                    self.viewer.sync()
                else:
                    # If the user closed the window, we must handle it
                    self.close() # Properly close the viewer resources
                    self.viewer = mujoco.viewer.launch(self.sim.model, self.sim.data) # And re-launch it
            except Exception:
                # This can happen if the viewer was closed abruptly
                self.viewer = mujoco.viewer.launch(self.sim.model, self.sim.data)
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

