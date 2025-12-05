import mujoco
import numpy as np
import time
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
from scipy.spatial.transform import Rotation as R
from dm_control import mjcf, mujoco as dm_mujoco
from dm_control.utils.inverse_kinematics import qpos_from_site_pose

class Simulation:
    def __init__(self, scene_path, output_dir=None):
        self.scene_path = scene_path
        self.output_dir = output_dir

        self.model = None
        self.data = None

        # --- Joint and actuator members ---
        self.arm_joint_names = []
        self.arm_joint_ids = []
        self.arm_act_ids = []
        self.hand_joint_ids = []
        self.hand_act_ids = []

        self.q_ref=[0.0, -1.51, -1.51, -1.51, 1.51, 0]

        # --- Vision members ---
        self.renderer = None
        self.cid = None
        self.intr = None

    def load(self):
        self.model = mujoco.MjModel.from_xml_path(self.scene_path)
        self.data = mujoco.MjData(self.model)

        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_lift_joint")
        qposadr = self.model.jnt_qposadr[jid]
        self.data.qpos[qposadr] = -math.pi/2
        mujoco.mj_forward(self.model, self.data)
    
    def ids_by_name(self, names, kind, role):
        out = []
        for n in names:
            i = mujoco.mj_name2id(self.model, kind, n)
            if i < 0: raise RuntimeError(f"Name not found in ids_by_name: {n}")
            out.append(i)
        if role == 'arm':
            self.arm_joint_names = names
            self.arm_joint_ids = out
        elif role == 'hand':
            self.hand_joint_ids = out

    def step(self, viewer, steps=100):
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            viewer.sync()

    def set_joint_positions(self, joint_ids, q_values):
        """
        Directly sets the position (angle) of a list of joints.
        
        Args:
            sim: The Simulation object.
            joint_ids: A list of joint IDs to modify.
            q_values: A list or numpy array of target joint angles.
        """
        if len(joint_ids) != len(q_values):
            raise ValueError("Mismatch between number of joint IDs and target angles.")
        
        for i, joint_id in enumerate(joint_ids):
            qpos_address = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_address] = q_values[i]
    
    #################################
    # --- Vision sensor handling ---#
    #################################

    def init_renderer(self, width=640, height=480):
        if self.renderer is None or self.renderer.width != width or self.renderer.height != height:
            self.renderer = mujoco.Renderer(self.model, width=int(width), height=int(height))

    def cam_id(self, name: str) -> int:
        self.cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if self.cid < 0:
            raise ValueError(f"Camera '{name}' not found.")

    def _compute_intrinsics(self, cam_id_int: int, width: int, height: int):
        fovy_deg = float(self.model.cam_fovy[cam_id_int])
        if fovy_deg <= 0:
            fovy_deg = 45.0
        fovy = np.deg2rad(fovy_deg)
        fy = (height / 2.0) / np.tan(fovy / 2.0)
        fx = fy * (width / float(height))
        cx = (width  - 1) / 2.0
        cy = (height - 1) / 2.0
        return dict(fx=fx, fy=fy, cx=cx, cy=cy, fovy_deg=fovy_deg, width=width, height=height)

    def _depth_to_vis(self, depth: np.ndarray, dmin: float = 0.25, dmax: float = 1.5) -> np.ndarray:
        d = np.clip(depth, dmin, dmax)
        d = (dmax - d) / max(1e-9, (dmax - dmin))
        d = (d * 255.0).astype(np.uint8)
        return d

    def _colorize_labels(self, labels: np.ndarray) -> np.ndarray:
        if labels is None:
            return None
        h, w = labels.shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        ids = np.unique(labels)
        for i in ids:
            if i < 0: 
                continue
            rng = np.random.default_rng(i)
            color = (rng.integers(32, 224, size=3)).astype(np.uint8)
            vis[labels == i] = color
        return vis

    def capture_rgb(self, width=640, height=480) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        self.init_renderer(width, height)
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.cid)
        return self.renderer.render()

    def capture_depth(self, width=640, height=480) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        self.init_renderer(width, height)
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.cid)
        depth = self.renderer.render().astype(np.float32)
        self.renderer.disable_depth_rendering()
        return depth

    def capture_labels(self, width=640, height=480) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        self.init_renderer(width, height)
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=self.cid)
        seg = self.renderer.render()
        self.renderer.disable_segmentation_rendering()
        if seg.ndim == 2:
            labels = seg
        else:
            labels = seg[:, :, 0]
        return labels.astype(np.int32)

    def get_frame(self, cam_name: str, width=640, height=480, depth_vis_range=(0.25, 1.5), do_seg=True):
        self.cam_id(cam_name)
        self.intr = self._compute_intrinsics(self.cid, width, height)
        rgb   = self.capture_rgb(width, height)
        depth = self.capture_depth(width, height)
        dvis  = self._depth_to_vis(depth, *depth_vis_range)
        labels = self.capture_labels(width, height) if do_seg else None
        svis   = self._colorize_labels(labels) if labels is not None else None
        return dict(
            cam_name=cam_name,
            t_wall=time.time(),
            rgb=rgb,
            depth=depth,
            depth_vis=dvis,
            labels=labels,
            seg_vis=svis,
            intrinsics=self.intr,
        )

    def save_frame(self, frame: dict, prefix="frame", outdir=None):
        import os
        outdir = outdir or os.getcwd()
        os.makedirs(outdir, exist_ok=True)
        base = os.path.join(outdir, prefix)
        try:
            import cv2
            cv2.imwrite(base + "_rgb.png", frame["rgb"][:, :, ::-1])
            cv2.imwrite(base + "_depth.png", frame["depth_vis"])
            if frame.get("seg_vis") is not None:
                cv2.imwrite(base + "_seg.png", frame["seg_vis"][:, :, ::-1])
        except Exception:
            np.save(base + "_rgb.npy", frame["rgb"])
            np.save(base + "_depth.npy", frame["depth"])
            if frame.get("labels") is not None:
                np.save(base + "_labels.npy", frame["labels"])

    def shutdown_renderer(self):
        try:
            if self.renderer is not None:
                self.renderer.close()
        except Exception:
            pass
        self.renderer = None

    ###########################
    # --- Gripper control --- #
    ###########################

    def set_gripper_angles(self, q_target):
        if len(q_target) != len(self.hand_act_ids):
            raise ValueError(f"q_target length ({len(q_target)}) does not match "
                             f"number of hand actuators ({len(self.hand_act_ids)}).")
        for i, act_id in enumerate(self.hand_act_ids):
            self.data.ctrl[act_id] = q_target[i]


    def move_gripper_to_angles(self, q_target, duration=1.0, viewer=None):
        """
        Moves the gripper to target angles by smoothly interpolating the control
        signals over the specified duration.
        """
        # 1. Get the starting joint angles of the gripper
        q_current = np.array([self.data.qpos[self.model.jnt_qposadr[j]]
                            for j in self.hand_joint_ids])
        
        # 2. Calculate the number of simulation steps
        dt = float(self.model.opt.timestep)
        steps = max(1, int(round(duration / dt)))

        # 3. Loop for the duration of the movement
        for k in range(steps):
            if viewer is not None and not viewer.is_running():
                return False

            # Calculate the blend factor, which goes from 0.0 to 1.0
            s = k / (steps - 1) if steps > 1 else 1.0
            
            # Calculate the intermediate target angles for this specific step
            q_interpolated = (1.0 - s) * q_current + s * q_target
            
            # Set the interpolated angles as the control signal for this step
            self.set_gripper_angles(q_interpolated)
            
            # Advance the simulation
            mujoco.mj_step(self.model, self.data)
            if viewer is not None:
                viewer.sync()
                
        return True

    ########################################
    # --- IK and path planning handling ---#
    ########################################

    def desired_qpos_from_ik(self, site_id,
                            target_pos, target_euler_xyz):
        physics = dm_mujoco.Physics.from_xml_path(self.scene_path)
        physics.data.qpos[:] = self.data.qpos

        quat_xyzw = R.from_euler('xyz', target_euler_xyz).as_quat()
        target_quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        site_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, site_id)

        # --- FIX: Use model.qpos0 instead of model.key_qpos[0] ---
        # This is more robust as it doesn't require a <key> to be defined in the XML.
        qpos_start = self.model.qpos0.copy() 
        
        for i, joint_id in enumerate(self.arm_joint_ids):
            qpos_addr = self.model.jnt_qposadr[joint_id]
            qpos_start[qpos_addr] = self.q_ref[i]
        
        physics.data.qpos[:] = qpos_start

        ik_result = qpos_from_site_pose(
            physics=physics,
            site_name=site_name,
            target_pos=target_pos,
            target_quat=target_quat_wxyz,
            joint_names=self.arm_joint_names,
            max_steps=200,
            tol=1e-3
        )

        return np.array([ik_result.qpos[self.model.jnt_qposadr[j]] for j in self.arm_joint_ids], dtype=float)

    def actuators_for_joints(self, role):
        act_ids = []
        joint_ids = self.arm_joint_ids if role == 'arm' else self.hand_joint_ids

        for j in joint_ids:
            found = None
            for a in range(self.model.nu):
                if self.model.actuator_trnid[a,0] == j:
                    found = a
                    break
            if found is None:
                raise RuntimeError(f"No actuator drives joint id {j} ({mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)})")
            act_ids.append(found)
        
        if role == 'arm':
            self.arm_act_ids = act_ids
        elif role == 'hand':
            self.hand_act_ids = act_ids
    
    ######################################
    # --- Path planning and execution ---#
    ######################################

    def arm_qpos(self):
        return np.array([self.data.qpos[self.model.jnt_qposadr[j]]
                        for j in self.arm_joint_ids], dtype=float)
    
    def execute_joint_path_pd(self, path, seg_T=0.4, hold_T=0.05, viewer=None):
        if path is None or len(path) == 0:
            return False

        dt = float(self.model.opt.timestep)
        steps_per_seg = max(2, int(round(seg_T / dt)))
        steps_hold    = max(1, int(round(hold_T / dt)))
        q_curr = self.arm_qpos().copy()

        for q_next in path:
            for k in range(steps_per_seg):
                if viewer is not None and not viewer.is_running():
                    return False
                s = self._cubic_blend(k / (steps_per_seg - 1))
                qdes = (1.0 - s) * q_curr + s * q_next
                self._set_ctrl_arm_qdes(qdes)
                mujoco.mj_step(self.model, self.data)
                if viewer is not None:
                    viewer.sync()
            q_curr = q_next.copy()
            for _ in range(steps_hold):
                if viewer is not None and not viewer.is_running():
                    return False
                self._set_ctrl_arm_qdes(q_curr)
                mujoco.mj_step(self.model, self.data)
                if viewer is not None:
                    viewer.sync()
        return True
    
    def _cubic_blend(self, s):
        s = max(0.0, min(1.0, s))
        return 3*s*s - 2*s*s*s
    
    def _set_ctrl_arm_qdes(self, qdes):
        for a, v in zip(self.arm_act_ids, qdes):
            self.data.ctrl[a] = float(v)

    def plan_straight_joint_path(self, q_goal, n_waypoints=80, shortest_wrap=True):
        q0 = self.arm_qpos()
        q1 = np.asarray(q_goal, float)
        if shortest_wrap:
            d = (q1 - q0 + np.pi) % (2*np.pi) - np.pi
            q_samples = np.array([q0 + s * d for s in np.linspace(0.0, 1.0, n_waypoints)])
        else:
            q_samples = np.linspace(q0, q1, n_waypoints)
        return q_samples
    
    def execute_straight_joint_move(self, q_goal, duration=2.0, n_waypoints=80, viewer=None):
        path = self.plan_straight_joint_path(q_goal, n_waypoints=n_waypoints, shortest_wrap=True)
        seg_T = float(duration) / max(1, (len(path) - 1))
        return self.execute_joint_path_pd(path, seg_T=seg_T, hold_T=0.05, viewer=viewer)