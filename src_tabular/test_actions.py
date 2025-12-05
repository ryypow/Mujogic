"""
Test script to visually see what each action does.
Tests the full cycle: THUMB_PUSH -> THUMB_RETRACT -> FINGER1_SWIPE -> FINGER1_RETRACT
"""
import time
from inhand_env import CanRotateEnv
from MinimalTranslator import MinimalTranslator

# Create environment with viewer
print("Starting environment with viewer...")
base_env = CanRotateEnv(render_mode="human")
env = MinimalTranslator(base_env)

# Reset to starting position
print("\nResetting to starting position...")
obs, _ = env.reset()
time.sleep(2)

print("\n" + "="*50)
print("TESTING 8 ACTIONS (Thumb + Finger3 curl/nudge + Finger2)")
print("="*50)

start_rot = base_env.get_object_z_rotation()
print(f"\nStarting Z rotation: {start_rot:.1f}°")
print(f"Target: -90°")

# Phase 1: THUMB_PUSH (action 1)
print("\n>>> Phase 1: THUMB_PUSH (action 1) - 10 steps")
for i in range(10):
    obs, reward, done, truncated, _ = env.step(1)
    z_rot = base_env.get_object_z_rotation()
    print(f"    Step {i+1}: Z = {z_rot:.1f}°")
    if done:
        print("    DROPPED!")
        break
time.sleep(1)

# Phase 2: THUMB_RETRACT (action 2)
print("\n>>> Phase 2: THUMB_RETRACT (action 2) - 5 steps")
for i in range(5):
    obs, reward, done, truncated, _ = env.step(2)
    z_rot = base_env.get_object_z_rotation()
    print(f"    Step {i+1}: Z = {z_rot:.1f}°")
    if done:
        print("    DROPPED!")
        break
time.sleep(1)

# Phase 3: FINGER3_CURL (action 3) - curl the tip first
print("\n>>> Phase 3: FINGER3_CURL (action 3) - 5 steps")
for i in range(5):
    obs, reward, done, truncated, _ = env.step(3)
    z_rot = base_env.get_object_z_rotation()
    print(f"    Step {i+1}: Z = {z_rot:.1f}°")
    if done:
        print("    DROPPED!")
        break
time.sleep(1)

# Phase 4: FINGER3_NUDGE (action 4) - sharp nudge with joint 8
print("\n>>> Phase 4: FINGER3_NUDGE (action 4) - 10 steps")
for i in range(10):
    obs, reward, done, truncated, _ = env.step(4)
    z_rot = base_env.get_object_z_rotation()
    print(f"    Step {i+1}: Z = {z_rot:.1f}°")
    if done:
        print("    DROPPED!")
        break
time.sleep(1)

# Phase 5: FINGER3_RETRACT (action 5)
print("\n>>> Phase 5: FINGER3_RETRACT (action 5) - 5 steps")
for i in range(5):
    obs, reward, done, truncated, _ = env.step(5)
    z_rot = base_env.get_object_z_rotation()
    print(f"    Step {i+1}: Z = {z_rot:.1f}°")
    if done:
        print("    DROPPED!")
        break
time.sleep(1)

# Phase 6: Another THUMB_PUSH cycle
print("\n>>> Phase 6: THUMB_PUSH again (action 1) - 10 steps")
for i in range(10):
    obs, reward, done, truncated, _ = env.step(1)
    z_rot = base_env.get_object_z_rotation()
    print(f"    Step {i+1}: Z = {z_rot:.1f}°")
    if done:
        print("    DROPPED!")
        break
time.sleep(1)

# Phase 7: FINGER2_PUSH (action 6)
print("\n>>> Phase 7: FINGER2_PUSH (action 6) - 10 steps")
for i in range(10):
    obs, reward, done, truncated, _ = env.step(6)
    z_rot = base_env.get_object_z_rotation()
    print(f"    Step {i+1}: Z = {z_rot:.1f}°")
    if done:
        print("    DROPPED!")
        break
time.sleep(1)

# Phase 8: FINGER2_RETRACT (action 7)
print("\n>>> Phase 8: FINGER2_RETRACT (action 7) - 5 steps")
for i in range(5):
    obs, reward, done, truncated, _ = env.step(7)
    z_rot = base_env.get_object_z_rotation()
    print(f"    Step {i+1}: Z = {z_rot:.1f}°")
    if done:
        print("    DROPPED!")
        break

final_rot = base_env.get_object_z_rotation()
print("\n" + "="*50)
print(f"STARTING:  {start_rot:.1f}°")
print(f"FINAL:     {final_rot:.1f}°")
print(f"ROTATED:   {final_rot - start_rot:.1f}°")
print(f"TARGET:    -90°")
print("="*50)

# Keep viewer open
print("\nPress Ctrl+C to exit...")
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    pass

env.close()
print("Done!")
