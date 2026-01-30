# inhand_env.py (Final Corrected Version)
import os
import numpy as np
import mujoco
import mujoco.viewer as mjv
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation
from simulation import Simulation

MAX_EPISODE_STEPS = 300

class CanRotateEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, target_degrees, render_mode=None):
        super(CanRotateEnv, self).__init__()

        # For tracking the cubes rotation progress
        self.cube_rotation_prev = None

        # Store initial finger positions (set in reset)
        self.initial_finger_positions = None

        self.target_rotation = np.deg2rad(target_degrees)
        self.target_tolerance = np.deg2rad(5)





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
        obs_size = len(self.sim.hand_joint_ids) + 7
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        self.render_mode = render_mode
        if render_mode == 'human':
            self.viewer = mjv.launch_passive(self.sim.model, self.sim.data) 
        else:
            self.viewer = None
        self.step_count = 0

    def _get_obs(self):
        finger_qpos = np.array([self.sim.data.qpos[self.sim.model.jnt_qposadr[j]] for j in self.sim.hand_joint_ids]) #
        obj_jnt_adr = self.sim.model.body_jntadr[self.obj_body_id] #
        obj_qpos_adr = self.sim.model.jnt_qposadr[obj_jnt_adr] #
        object_pose = self.sim.data.qpos[obj_qpos_adr : obj_qpos_adr + 7] 
        return np.concatenate([finger_qpos, object_pose])

    def _get_finger_positions(self):
        """Get current finger joint positions."""
        return np.array([self.sim.data.qpos[self.sim.model.jnt_qposadr[j]]
                        for j in self.sim.hand_joint_ids])

    def _calculate_reward(self, cube_rotation_new, cube_rotation_prev):
        TARGET_ROTATION = self.target_rotation
        TARGET_TOLERANCE = self.target_tolerance

        # Current rotation state
        obj_vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.sim.model, self.sim.data, mujoco.mjtObj.mjOBJ_BODY, self.obj_body_id, obj_vel, 0)
        angular_velocity_z = obj_vel[2]

        # Distance to target rotation
        current_distance = abs(TARGET_ROTATION - cube_rotation_new)

        # Progress reward - reward for getting closer to goal
        if cube_rotation_prev is not None:
            previous_distance = abs(TARGET_ROTATION - cube_rotation_prev)
            progress_reward = (previous_distance - current_distance) * 10.0
        else:
            progress_reward = 0.0

        # Bonus for reaching target
        if current_distance < TARGET_TOLERANCE:
            nearTarget_bonus = 100.0
        else:
            nearTarget_bonus = 0.0

        # Penalty for spinning fast near target
        if current_distance < np.deg2rad(15):
            nearTarget_velocity_penalty = abs(angular_velocity_z) * 0.5
        else:
            nearTarget_velocity_penalty = 0.0

        # Rotation speed reward (encourage rotation in correct direction)
        rotation_reward = angular_velocity_z * 5.0

        # Distance from palm (survival reward)
        can_pos = self.sim.data.xpos[self.obj_body_id]
        palm_pos = self.sim.data.site_xpos[self.site_id]
        distance_from_palm = np.linalg.norm(can_pos - palm_pos)
        survival_reward = 0.1 - distance_from_palm

        # --- Pose Deviation Penalty (NEW) ---
        # Penalize fingers that drift too far from initial position
        # Only penalize if fingers drift significantly (threshold-based)
        pose_penalty = 0.0
        if self.initial_finger_positions is not None:
            current_positions = self._get_finger_positions()
            pose_diff = np.sum((current_positions - self.initial_finger_positions) ** 2)
            # Only apply penalty if deviation is large (allow some movement)
            if pose_diff > 0.5:  # Threshold before penalty kicks in
                pose_penalty = (pose_diff - 0.5) * 0.1  # Reduced scale factor

        # --- Contact Reward ---
        contact_reward = 0.0
        fingers_in_contact = set()

        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            is_geom1_fingertip = geom1 in self.fingertip_geom_ids
            is_geom2_fingertip = geom2 in self.fingertip_geom_ids
            is_geom1_can = geom1 == self.can_geom_id
            is_geom2_can = geom2 == self.can_geom_id

            if (is_geom1_fingertip and is_geom2_can):
                fingers_in_contact.add(geom1)
            elif (is_geom2_fingertip and is_geom1_can):
                fingers_in_contact.add(geom2)

        if len(fingers_in_contact) >= 3:
            contact_reward = 5.0
        elif len(fingers_in_contact) > 0:
            contact_reward = 0.5 * len(fingers_in_contact)

        # Combine all reward components
        total_reward = (progress_reward + nearTarget_bonus + rotation_reward +
                       survival_reward + contact_reward -
                       nearTarget_velocity_penalty - pose_penalty)
        return total_reward

#---- need to update for different target positions
    def _is_terminated(self, z_rotation):
        can_z_pos = self.sim.data.xpos[self.obj_body_id][2] 
        palm_z_pos = self.sim.data.site_xpos[self.site_id][2] 
        
        #Terminate if target position achieved
        TARGET_ROTATION = self.target_rotation #target rotation passed into canRotateEnv
        TARGET_TOLERANCE = self.target_tolerance #5 deg tolerance

        #If target achieved, terminated = true
        #if abs(TARGET_ROTATION - z_rotation) < TARGET_TOLERANCE:
         #   return True
        
        #if cube dropped or max steps exceeded, terminate = true
        #if can_z_pos < (palm_z_pos - 0.05) or self.step_count > MAX_EPISODE_STEPS:
         #   return True
        
        return False

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

        palm_surface_pos = self.sim.data.site_xpos[self.site_id].copy()
        # Original working cube position
        object_start_pos = palm_surface_pos + np.array([0.011, -0.03, 0.075])
        obj_jnt_adr = self.sim.model.body_jntadr[self.obj_body_id]
        obj_qpos_adr = self.sim.model.jnt_qposadr[obj_jnt_adr]
        self.sim.data.qpos[obj_qpos_adr : obj_qpos_adr + 3] = object_start_pos
        self.sim.data.qpos[obj_qpos_adr + 3 : obj_qpos_adr + 7] = [1, 0, 0, 0]

        mujoco.mj_forward(self.sim.model, self.sim.data)

        for _ in range(20):
            mujoco.mj_step(self.sim.model, self.sim.data)

        # Original working finger positions
        q_open_angles = np.array([
            0.8, 0.8, 0.8, 0.8,  # Finger 1 - light contact
            0.8, 0.8, 0.8, 0.8,  # Finger 2 - light contact
            0.9, 0.8, 0.9, 0.8,  # Finger 3
            0.8, 0.8, 0.8, 0.8   # Thumb
        ])
        self.sim.set_joint_positions(self.sim.hand_joint_ids, q_open_angles)
        for i, act_id in enumerate(self.sim.hand_act_ids):
            self.sim.data.ctrl[act_id] = q_open_angles[i]

        mujoco.mj_forward(self.sim.model, self.sim.data)

        # Store initial finger positions for pose penalty
        self.initial_finger_positions = self._get_finger_positions().copy()

        # Reset rotation tracking
        self.cube_rotation_prev = None

        return self._get_obs(), {}

    def step(self, action):
        target_angles = np.array([self.sim.data.qpos[self.sim.model.jnt_qposadr[j]] for j in self.sim.hand_joint_ids]) + action
        self.sim.move_gripper_to_angles(target_angles, 0.5) #

        if self.render_mode != "headless" and self.viewer: #only syncs when in human mode
            self.viewer.sync()        

        self.step_count += 1
        
        observation = self._get_obs()

        #Get cubes orientation, get euler angle via scipy.Rotation
        cube_quart = observation[-4:]
        r = Rotation.from_quat(cube_quart) #reminder: working with radians
        euler = r.as_euler('xyz') #x,y,z coordinates
        z_rotation = euler[2]

        #UPDATED: passing the new cube rotation and the rotation history into reward function
        reward = self._calculate_reward(z_rotation, self.cube_rotation_prev)
        terminated = self._is_terminated(z_rotation)
        truncated = self.step_count >= MAX_EPISODE_STEPS

        #update the previous z-rotation
        self.cube_rotation_prev = z_rotation        


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


        #logic to capture PIL snaps for termination/truncation situations to observe cube
        elif self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.sim.model, height=480, width=640)
            renderer.update_scene(self.sim.data)
            frame = renderer.render()
            renderer.close()
            return frame
        
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
