# inhand_env.py (Final Corrected Version)
import os
import numpy as np
import mujoco
import mujoco.viewer as mjv
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from simulation import Simulation

MAX_EPISODE_STEPS = 300
TARGET_ROTATION = -90.0
TARGET_TOLERANCE = 5.0

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
        self.action_space = spaces.Box(low=-0.03, high=0.03, shape=(16,), dtype=np.float32)
        obs_size = len(self.sim.hand_joint_ids) + 4 #16 joints, 3 position, and 1 (remaining rotation)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        self.render_mode = render_mode
        if render_mode == 'human':
            self.viewer = mjv.launch_passive(self.sim.model, self.sim.data) 
        else:
            self.viewer = None
        self.step_count = 0
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

    def get_rotation_achieved(self):
        """how many degrees rotated from start"""
        current = self.get_object_z_rotation()
        diff = current - self.initial_z_rotation

        if diff > 180.0:
            diff -= 360
        elif diff < -180.0:
            diff += 360.0

        return abs(diff)


    def _get_obs(self):
        #finger positions
        finger_qpos = np.array([self.sim.data.qpos[self.sim.model.jnt_qposadr[j]] for j in self.sim.hand_joint_ids]) #

        #object position
        obj_pos = self.sim.data.xpos[self.obj_body_id]
        palm_pos = self.sim.data.site_xpos[self.site_id]
        real_pos = obj_pos - palm_pos

        rotation_achieved = self.get_rotation_achieved()
        remaining_rotation = (90.0 - rotation_achieved) / 90.0 #[1,0] -> will approach 0 as progress is made
        remaining = np.clip(remaining_rotation, -1.0, 1.0) #clip if it overshoots

        #raw rotation
        #z_rot_normalized = self.get_object_z_rotation() / 180.0 [-1, 1]
        #goal progress
        #goal_progress = (TARGET_ROTATION - self.get_object_z_rotation()) / 90.0 #normalized

        return np.concatenate([finger_qpos, real_pos, [remaining]]).astype(np.float32)


    def _calculate_reward(self, z_rotat_new, z_rot_prev):
        #survival reward
        can_pos = self.sim.data.xpos[self.obj_body_id]
        palm_pos = self.sim.data.site_xpos[self.site_id]
        distance_from_palm = np.linalg.norm(can_pos - palm_pos)
        survival_reward = 0.1 - distance_from_palm

        if self.sim.data.xpos[self.obj_body_id][2] < 0.3:
            survival_reward = -10.0

        #Progress toward goal
        dist_new = abs(TARGET_ROTATION -z_rotat_new)
        dist_prev = abs(TARGET_ROTATION - z_rot_prev)
        progress_reward = (dist_prev - dist_new) #favors positive value

        # Combine all reward components
        total_reward = survival_reward + progress_reward  #+ rotation_reward 
        return total_reward

    def _is_terminated(self):
        dropped = self.sim.data.xpos[self.obj_body_id][2] < 0.3
        success = self.get_rotation_achieved() >= (90.0 - TARGET_TOLERANCE)
        return dropped or success

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        
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

        q_open_angles = np.array([1.0, 0.3, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.3, 1.0, 1.3, 1.0, 0.8, 1.3, 0.8, 0.5]) #
        self.sim.set_joint_positions(self.sim.hand_joint_ids, q_open_angles) #
        for i, act_id in enumerate(self.sim.hand_act_ids):
            self.sim.data.ctrl[act_id] = q_open_angles[i] #

        mujoco.mj_forward(self.sim.model, self.sim.data)

        if self.render_mode != "headless":
            self.viewer.sync()

        self.initial_z_rotation = self.get_object_z_rotation()

        return self._get_obs(), {}

    def step(self, action):
        z_rot_prev = self.get_object_z_rotation()

        target_angles = np.array([self.sim.data.qpos[self.sim.model.jnt_qposadr[j]] for j in self.sim.hand_joint_ids]) + action
        self.sim.move_gripper_to_angles(target_angles, 0.5) #

        if self.render_mode != "headless":
            self.viewer.sync()

        self.step_count += 1
        
        z_rot_new = self.get_object_z_rotation()
        observation = self._get_obs()
        reward = self._calculate_reward(z_rot_new, z_rot_prev)
        terminated = self._is_terminated()
        truncated = self.step_count >= MAX_EPISODE_STEPS

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
