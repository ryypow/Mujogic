import mujoco
import mujoco.viewer
import numpy as np
import time
import os
from scipy.spatial.transform import Rotation as R

# Import the simulation class from the provided file
from simulation import Simulation #

# --- Helper Function ---
def get_object_z_rotation(sim, obj_body_id):
    """
    Calculates the Z-axis rotation of an object from its quaternion.
    """
    # Get the quaternion (w, x, y, z) from MuJoCo data
    quat_wxyz = sim.data.xquat[obj_body_id]
    
    # Scipy's Rotation object expects (x, y, z, w)
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    
    # Create a Rotation object
    r = R.from_quat(quat_xyzw)
    
    # Convert to Euler angles (xyz order) in degrees
    euler_angles_deg = r.as_euler('xyz', degrees=True)
    
    # Return the Z-axis rotation
    return euler_angles_deg[2]

def print_object_status(sim, obj_body_id):
    """Prints the object's current position and Z rotation."""
    
    # We must call mj_forward() to ensure all physics-derived
    # values (like xpos and xquat) are up-to-date.
    mujoco.mj_forward(sim.model, sim.data)
            
    # Get object position
    obj_pos = sim.data.xpos[obj_body_id]
    
    # Get object Z rotation
    obj_z_rot = get_object_z_rotation(sim, obj_body_id)
    
    # Print to console
    print(f"  > Object Position (x, y, z):  ({obj_pos[0]:.4f}, {obj_pos[1]:.4f}, {obj_pos[2]:.4f})")
    print(f"  > Object Z Rotation (degrees): {obj_z_rot:.2f}°")


# --- Main Script ---
def main():
    print("Loading simulation...")
    
    # --- 1. Initialize Simulation ---
    scene_path = os.path.join(os.path.dirname(__file__), "scene.xml") #
    sim = Simulation(scene_path=scene_path)
    sim.load()

    # --- 2. Get Object and Joint IDs ---
    try:
        obj_body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "obj1")
        site_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        if obj_body_id == -1 or site_id == -1:
            raise ValueError()
    except ValueError:
        print("Error: Could not find 'obj1' body or 'attachment_site' site.")
        return

    # --- 3. Get Hand and Arm Joint/Actuator IDs ---
    # This setup is copied from inhand_env.py to match the RL environment
    sim.ids_by_name([
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
    ], mujoco.mjtObj.mjOBJ_JOINT, 'arm') #
    
    sim.ids_by_name([
        "1", "0", "2", "3", "5", "4", "6", "7", 
        "9", "8", "10", "11", "12", "13", "14", "15"
    ], mujoco.mjtObj.mjOBJ_JOINT, 'hand') #
    
    sim.actuators_for_joints('arm') #
    sim.actuators_for_joints('hand') #
    print(f"Found {len(sim.hand_act_ids)} hand actuators.")

    # --- 4. Set Initial Scene State ---
    # This mimics the reset() function from the environment
    mujoco.mj_resetData(sim.model, sim.data)
    
    # --- THIS SECTION IS UPDATED AS PER YOUR REQUEST ---
    print("Calculating IK for upward-facing palm pose...")
    target_pos_up = np.array([0.4, 0.0, 0.5])
    target_euler_up = np.array([0, 0, 0])
    
    # Use the IK function from simulation.py
    q_palm_up = sim.desired_qpos_from_ik(site_id, target_pos_up, target_euler_up) #
    print("IK calculation complete. Setting arm pose.")
    
    # Set the joint positions and controls
    sim.set_joint_positions(sim.arm_joint_ids, q_palm_up) #
    for i, act_id in enumerate(sim.arm_act_ids):
        sim.data.ctrl[act_id] = q_palm_up[i] #
    # --- END OF UPDATED SECTION ---
    
    mujoco.mj_forward(sim.model, sim.data)

    # Set object pose in palm
    palm_surface_pos = sim.data.site_xpos[site_id].copy() #
    object_start_pos = palm_surface_pos + np.array([0.011, -0.03, 0.075]) #
    obj_jnt_adr = sim.model.body_jntadr[obj_body_id] #
    obj_qpos_adr = sim.model.jnt_qposadr[obj_jnt_adr] #
    sim.data.qpos[obj_qpos_adr : obj_qpos_adr + 3] = object_start_pos #
    sim.data.qpos[obj_qpos_adr + 3 : obj_qpos_adr + 7] = [1, 0, 0, 0] # Identity quaternion

    mujoco.mj_forward(sim.model, sim.data)

    # --- 5. Define Student-Editable Control Actions ---
    
    # This is the "open hand" pose from the environment's reset function
    q_open_angles = np.array([
        1.0, 0.3, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 
        1.3, 1.0, 1.3, 1.0, 0.8, 1.3, 0.8, 0.5
    ]) #
    
    # This is a "closed hand" pose (students can edit these values!)
    q_close_angles = np.array([
        1.0, 2.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 
        2.0, 1.0, 2.0, 1.0, 0.8, 2.0, 0.8, 0.5
    ])

    print("\n✅ Launching viewer.")
    print("✅ The code will now send open/close commands to the hand.")
    print("✅ Students: Try editing 'q_open_angles' and 'q_close_angles'!")
    print("-----------------------------------------------------------")

    # --- 6. Launch the Viewer and Start Control Loop ---
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        
        # Start with the hand open
        print("Setting initial open pose...")
        sim.move_gripper_to_angles(q_open_angles, duration=1.0, viewer=viewer) #
        print_object_status(sim, obj_body_id)
        time.sleep(1.0) # Pause to observe

        # Main control loop
        while viewer.is_running():
            
            # --- ACTION 1: CLOSE HAND ---
            print("\n--- Sending CLOSE command ---")
            
            # This function sends the control signal and steps the simulation
            # for the specified duration, syncing the viewer as it goes.
            sim.move_gripper_to_angles(q_close_angles, duration=1.5, viewer=viewer) #
            
            # Print the state *after* the action is complete
            print_object_status(sim, obj_body_id)
            
            # Pause for observation
            time.sleep(2.0)

            # Check if viewer is still running before next action
            if not viewer.is_running():
                break

            # --- ACTION 2: OPEN HAND ---
            print("\n--- Sending OPEN command ---")
            
            # Send the "open" command
            sim.move_gripper_to_angles(q_open_angles, duration=1.5, viewer=viewer) #
            
            # Print the state *after* the action is complete
            print_object_status(sim, obj_body_id)
            
            # Pause for observation
            time.sleep(2.0)

if __name__ == "__main__":
    main()