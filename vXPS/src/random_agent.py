import time
from inhand_env import CanRotateEnv
import imageio

EPISODES_TO_RUN = 20

#load simulation environment
env = CanRotateEnv(render_mode='human')

#output file to collect current step, positions, and orientations
output_file = open("actions_location_orientation.txt", 'w')

#init  video writer
video_writer = imageio.get_writer('random_agent_demo.mp4', fps=30)
env.sim.init_renderer(width=640,height=480)
env.sim.cam_id("camera")

# --- Run the evaluation ---
for episode in range(EPISODES_TO_RUN):
    output_file.write(f"\n--- Starting Episode {episode + 1} ---")
    
    #Reset the environment
    obs, info = env.reset()
    
    terminated = False
    truncated = False
    total_reward = 0
    step_count = 0

    while not (terminated or truncated):
        
        action = env.action_space.sample() #random sample from action space
        
        #log action
        output_file.write(f'Step {step_count}: Action = {action}')

        #capture frame
        frame = env.sim.capture_rgb(width=640, height=480)
        video_writer.append_data(frame)

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.05)
        
        # Extract and print object position/orientation
        object_pose = obs[-7:]
        position = object_pose[:3]
        quaternion = object_pose[3:]
        
        output_file.write(f"  Position: x={position[0]:.3f}, y={position[1]:.3f}, z={position[2]:.3f}")
        output_file.write(f"  Quaternion: {quaternion}\n")
        
        total_reward += reward
        step_count += 1
               
        #time.sleep(1/60) # Keep visualization smooth

    output_file.write(f"Episode {episode + 1} finished. Total Reward: {total_reward:.2f}\n")

output_file.close()
video_writer.close()
env.close()
print("\nEvaluation finished.")

