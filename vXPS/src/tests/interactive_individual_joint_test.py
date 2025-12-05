"""
Interactive individual joint test - manually set individual joint angles.
Supports notation like 'finger1.2 0.5' for finger 1, joint 2.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import mujoco.viewer
import numpy as np
import re
from simulation import Simulation

# Joint mapping for reference
JOINT_MAP = {
    'finger1': {'range': (0, 4), 'joints': {1: 0, 2: 1, 3: 2, 4: 3}},  # mcp, pip, dip, tip
    'finger2': {'range': (4, 8), 'joints': {1: 4, 2: 5, 3: 6, 4: 7}},
    'finger3': {'range': (8, 12), 'joints': {1: 8, 2: 9, 3: 10, 4: 11}},
    'thumb': {'range': (12, 16), 'joints': {1: 12, 2: 13, 3: 14, 4: 15}},
}

# Aliases for convenience
JOINT_MAP['f1'] = JOINT_MAP['finger1']
JOINT_MAP['f2'] = JOINT_MAP['finger2']
JOINT_MAP['f3'] = JOINT_MAP['finger3']
JOINT_MAP['t'] = JOINT_MAP['thumb']

JOINT_NAMES = ['mcp', 'pip', 'dip', 'tip']


def parse_command(cmd):
    """
    Parse commands like:
      finger1.2 0.5    -> set finger1 joint 2 to 0.5
      f1.2 0.5         -> same (alias)
      finger1 0.5      -> set all finger1 joints to 0.5
      set 5 0.5        -> set joint index 5 to 0.5
      all 0.5          -> set all joints to 0.5
    Returns: (action, indices, value) or (action, None, None) for commands like 'show', 'reset'
    """
    cmd = cmd.strip().lower()

    if cmd in ('quit', 'q'):
        return ('quit', None, None)
    elif cmd == 'show':
        return ('show', None, None)
    elif cmd == 'reset':
        return ('reset', None, None)
    elif cmd == 'help':
        return ('help', None, None)

    # Check for "all <value>"
    if cmd.startswith('all '):
        try:
            val = float(cmd.split()[1])
            return ('set', list(range(16)), val)
        except (ValueError, IndexError):
            return ('error', None, 'Invalid value for "all" command')

    # Check for "set <index> <value>"
    if cmd.startswith('set '):
        parts = cmd.split()
        try:
            idx = int(parts[1])
            val = float(parts[2])
            if 0 <= idx < 16:
                return ('set', [idx], val)
            else:
                return ('error', None, f'Index {idx} out of range (0-15)')
        except (ValueError, IndexError):
            return ('error', None, 'Usage: set <index> <value>')

    # Check for finger/thumb commands with optional joint specifier
    # Pattern: (finger1|finger2|finger3|thumb|f1|f2|f3|t)[.joint_num] value
    pattern = r'^(finger[123]|thumb|f[123]|t)(?:\.(\d))?(?:\s+(.+))?$'
    match = re.match(pattern, cmd)

    if match:
        finger_name = match.group(1)
        joint_num = match.group(2)
        value_str = match.group(3)

        if value_str is None:
            return ('error', None, f'Please provide a value for {finger_name}')

        try:
            val = float(value_str)
        except ValueError:
            return ('error', None, f'Invalid value: {value_str}')

        finger_info = JOINT_MAP.get(finger_name)
        if not finger_info:
            return ('error', None, f'Unknown finger: {finger_name}')

        if joint_num:
            # Individual joint
            joint_idx = int(joint_num)
            if joint_idx < 1 or joint_idx > 4:
                return ('error', None, f'Joint number must be 1-4, got {joint_idx}')
            actual_idx = finger_info['joints'][joint_idx]
            return ('set', [actual_idx], val)
        else:
            # All joints in finger
            start, end = finger_info['range']
            return ('set', list(range(start, end)), val)

    return ('error', None, f'Unknown command: {cmd}')


def print_help():
    print("\n" + "=" * 70)
    print("INTERACTIVE INDIVIDUAL JOINT CONTROLLER")
    print("=" * 70)
    print("\nJoint mapping (16 joints total):")
    print("  Finger 1: joints 0-3  (index 0=mcp, 1=pip, 2=dip, 3=tip)")
    print("  Finger 2: joints 4-7  (index 4=mcp, 5=pip, 6=dip, 7=tip)")
    print("  Finger 3: joints 8-11 (index 8=mcp, 9=pip, 10=dip, 11=tip)")
    print("  Thumb:    joints 12-15")
    print("\n" + "-" * 70)
    print("COMMANDS:")
    print("-" * 70)
    print("\n  INDIVIDUAL JOINT (use notation: finger.joint_number)")
    print("    finger1.1 <val>  - Set finger 1, joint 1 (mcp) to value")
    print("    finger1.2 <val>  - Set finger 1, joint 2 (pip) to value")
    print("    finger1.3 <val>  - Set finger 1, joint 3 (dip) to value")
    print("    finger1.4 <val>  - Set finger 1, joint 4 (tip) to value")
    print("    f1.2 <val>       - Short form: f1, f2, f3 for fingers, t for thumb")
    print("    thumb.1 <val>    - Set thumb joint 1")
    print("    t.3 <val>        - Short form for thumb joint 3")
    print("\n  WHOLE FINGER")
    print("    finger1 <val>    - Set all 4 joints of finger 1")
    print("    finger2 <val>    - Set all 4 joints of finger 2")
    print("    finger3 <val>    - Set all 4 joints of finger 3")
    print("    thumb <val>      - Set all 4 thumb joints")
    print("    f1/f2/f3/t <val> - Short forms")
    print("\n  DIRECT INDEX")
    print("    set <idx> <val>  - Set joint by index (0-15)")
    print("\n  ALL JOINTS")
    print("    all <val>        - Set all 16 joints to value")
    print("\n  OTHER")
    print("    show             - Display current joint angles")
    print("    reset            - Reset all joints to 0")
    print("    help             - Show this help")
    print("    quit / q         - Exit")
    print("\n" + "-" * 70)
    print("EXAMPLES:")
    print("-" * 70)
    print("  > finger1.2 0.8    # Set finger 1 pip joint to 0.8")
    print("  > f2.1 1.0         # Set finger 2 mcp joint to 1.0")
    print("  > thumb 0.5        # Set all thumb joints to 0.5")
    print("  > t.3 0.3          # Set thumb joint 3 to 0.3")
    print("  > set 5 0.7        # Set joint index 5 to 0.7")
    print("  > all 0.0          # Reset all to 0")
    print("=" * 70)


def print_angles_detailed(hand_angles):
    """Print current angles in a detailed format."""
    print("\nCurrent joint angles:")
    print("-" * 50)
    for finger_name, info in [('Finger 1', JOINT_MAP['finger1']),
                               ('Finger 2', JOINT_MAP['finger2']),
                               ('Finger 3', JOINT_MAP['finger3']),
                               ('Thumb', JOINT_MAP['thumb'])]:
        start, end = info['range']
        angles = hand_angles[start:end]
        formatted = [f"{JOINT_NAMES[i]}={angles[i]:.3f}" for i in range(4)]
        print(f"  {finger_name:10s}: {', '.join(formatted)}")
    print("-" * 50)


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
    palm_pos = sim.data.site_xpos[site_id].copy()
    cube_pos = palm_pos + np.array([0.0, 0.0, 0.05])

    obj_body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "obj1")
    obj_joint_id = sim.model.body_jntadr[obj_body_id]
    obj_qpos_addr = sim.model.jnt_qposadr[obj_joint_id]

    sim.data.qpos[obj_qpos_addr:obj_qpos_addr+3] = cube_pos
    sim.data.qpos[obj_qpos_addr+3:obj_qpos_addr+7] = [1, 0, 0, 0]

    obj_qvel_addr = sim.model.jnt_dofadr[obj_joint_id]
    sim.data.qvel[obj_qvel_addr:obj_qvel_addr+6] = 0

    mujoco.mj_forward(sim.model, sim.data)
    print(f"Cube placed at palm position: {cube_pos}")

    # Current hand angles - start in grasp position matching inhand_env.py reset()
    hand_angles = np.array(  [1.0, -0.3, 0.7, 0.7,  # Finger 1 (open)
   1.0, 0.8, 0.8, 0.8,   # Finger 2 (open)
   1.3, 1.5, 1.0, 1.4,   # Finger 3 (open)
   -0.3, 1.4, 1.0, 2.0])

    # Apply initial grasp position
    sim.set_joint_positions(sim.hand_joint_ids, hand_angles)
    for i, act_id in enumerate(sim.hand_act_ids):
        sim.data.ctrl[act_id] = hand_angles[i]
    mujoco.mj_forward(sim.model, sim.data)

    print_help()

    # Launch viewer
    viewer = mujoco.viewer.launch_passive(sim.model, sim.data)

    while viewer.is_running():
        try:
            cmd = input("\n> ").strip()

            if not cmd:
                continue

            action, indices, value = parse_command(cmd)

            if action == 'quit':
                break
            elif action == 'show':
                print_angles_detailed(hand_angles)
                continue
            elif action == 'reset':
                hand_angles = np.zeros(16)
                print("Reset all joints to 0")
            elif action == 'help':
                print_help()
                continue
            elif action == 'error':
                print(f"Error: {value}")
                continue
            elif action == 'set':
                for idx in indices:
                    hand_angles[idx] = value
                if len(indices) == 1:
                    print(f"Set joint {indices[0]} to {value}")
                else:
                    print(f"Set joints {indices} to {value}")

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
