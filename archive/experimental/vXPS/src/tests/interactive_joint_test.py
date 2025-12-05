"""
Interactive joint test - manually set angles and observe.
The simulation stays open and you can change values in the console.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import mujoco.viewer
import numpy as np
from simulation import Simulation

def main():
    # Load simulation
    scene_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scene.xml")
    sim = Simulation(scene_path=scene_path)
    sim.load()

    # Get IDs
    site_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    sim.ids_by_name(["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                     "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
                    mujoco.mjtObj.mjOBJ_JOINT, 'arm')
    sim.ids_by_name(["1", "0", "2", "3", "5", "4", "6", "7",
                     "9", "8", "10", "11", "12", "13", "14", "15"],
                    mujoco.mjtObj.mjOBJ_JOINT, 'hand')
    sim.actuators_for_joints('arm')
    sim.actuators_for_joints('hand')

    # Set arm to palm-up position
    mujoco.mj_resetData(sim.model, sim.data)
    target_pos = np.array([0.4, 0.0, 0.5])
    target_euler = np.array([0, 0, 0])
    q_arm = sim.desired_qpos_from_ik(site_id, target_pos, target_euler)
    sim.set_joint_positions(sim.arm_joint_ids, q_arm)
    for i, act_id in enumerate(sim.arm_act_ids):
        sim.data.ctrl[act_id] = q_arm[i]
    mujoco.mj_forward(sim.model, sim.data)

    # Place cube in the palm
    # Get the palm position from the attachment site
    palm_pos = sim.data.site_xpos[site_id].copy()
    # Offset slightly above the palm surface
    cube_pos = palm_pos + np.array([0.0, 0.0, 0.05])

    # Get the cube's free joint and set its position
    obj_body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "obj1")
    obj_joint_id = sim.model.body_jntadr[obj_body_id]
    obj_qpos_addr = sim.model.jnt_qposadr[obj_joint_id]

    # Free joint uses 7 values: x, y, z, qw, qx, qy, qz
    sim.data.qpos[obj_qpos_addr:obj_qpos_addr+3] = cube_pos
    sim.data.qpos[obj_qpos_addr+3:obj_qpos_addr+7] = [1, 0, 0, 0]  # Identity quaternion

    # Zero out velocities
    obj_qvel_addr = sim.model.jnt_dofadr[obj_joint_id]
    sim.data.qvel[obj_qvel_addr:obj_qvel_addr+6] = 0

    mujoco.mj_forward(sim.model, sim.data)
    print(f"Cube placed at palm position: {cube_pos}")

    # Current hand angles (start at zero)
    hand_angles = np.zeros(16)

    print("=" * 60)
    print("INTERACTIVE JOINT CONTROLLER")
    print("=" * 60)
    print("\nJoint mapping:")
    print("  [0-3]   Finger 1: indices 0(mcp), 1(pip), 2(dip), 3(tip)")
    print("  [4-7]   Finger 2: indices 4(mcp), 5(pip), 6(dip), 7(tip)")
    print("  [8-11]  Finger 3: indices 8(mcp), 9(pip), 10(dip), 11(tip)")
    print("  [12-15] Thumb:    indices 12, 13, 14, 15")
    print("\nCommands:")
    print("  set <index> <value>  - Set single joint (e.g., 'set 0 1.0')")
    print("  finger1 <value>      - Set all finger 1 joints")
    print("  finger2 <value>      - Set all finger 2 joints")
    print("  finger3 <value>      - Set all finger 3 joints")
    print("  thumb <value>        - Set all thumb joints")
    print("  all <value>          - Set all joints")
    print("  show                 - Show current angles")
    print("  reset                - Reset all to 0")
    print("  quit                 - Exit")
    print("=" * 60)

    # Launch viewer
    viewer = mujoco.viewer.launch_passive(sim.model, sim.data)

    while viewer.is_running():
        try:
            cmd = input("\n> ").strip().lower()

            if cmd == "quit" or cmd == "q":
                break
            elif cmd == "show":
                print(f"Current angles: {hand_angles}")
            elif cmd == "reset":
                hand_angles = np.zeros(16)
                print("Reset all to 0")
            elif cmd.startswith("set "):
                parts = cmd.split()
                idx = int(parts[1])
                val = float(parts[2])
                hand_angles[idx] = val
                print(f"Set joint {idx} to {val}")
            elif cmd.startswith("finger1 "):
                val = float(cmd.split()[1])
                hand_angles[0:4] = val
                print(f"Set finger 1 (0-3) to {val}")
            elif cmd.startswith("finger2 "):
                val = float(cmd.split()[1])
                hand_angles[4:8] = val
                print(f"Set finger 2 (4-7) to {val}")
            elif cmd.startswith("finger3 "):
                val = float(cmd.split()[1])
                hand_angles[8:12] = val
                print(f"Set finger 3 (8-11) to {val}")
            elif cmd.startswith("thumb "):
                val = float(cmd.split()[1])
                hand_angles[12:16] = val
                print(f"Set thumb (12-15) to {val}")
            elif cmd.startswith("all "):
                val = float(cmd.split()[1])
                hand_angles[:] = val
                print(f"Set all joints to {val}")
            else:
                print("Unknown command. Type 'quit' to exit.")
                continue

            # Apply angles
            sim.set_joint_positions(sim.hand_joint_ids, hand_angles)
            for i, act_id in enumerate(sim.hand_act_ids):
                sim.data.ctrl[act_id] = hand_angles[i]

            # Step physics and update viewer
            for _ in range(50):
                mujoco.mj_step(sim.model, sim.data)
                viewer.sync()

        except (ValueError, IndexError) as e:
            print(f"Error: {e}")
        except EOFError:
            break

    viewer.close()
    print("Done!")

if __name__ == "__main__":
    main()
