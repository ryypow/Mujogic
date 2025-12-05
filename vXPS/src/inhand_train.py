# inhand_train.py (Student Skeleton)
import os
import numpy as np
import time
from inhand_env import CanRotateEnv 
# --- TODO: Import your agent class ---
# from agent import MyRLAgent  # e.g., PPOAgent

# Create a directory to save logs and models
log_dir = "my_agent_logs/"
os.makedirs(log_dir, exist_ok=True)

# --- Configuration ---
TOTAL_TIMESTEPS = 8_000_000
STEPS_PER_COLLECT = 2048  # How many steps to run per "collect" phase
LEARNING_RATE = 3e-4
DEVICE = 'cpu' # 'cuda' or 'cpu'

# --- TODO: Initialize the Environment ---
env = CanRotateEnv(render_mode="headless")
print(f"Observation space: {env.observation_space.shape}")
print(f"Action space: {env.action_space.shape}")

# --- TODO: Initialize your Agent ---
# agent = MyRLAgent(
#     obs_space_shape=env.observation_space.shape,
#     action_space_shape=env.action_space.shape,
#     learning_rate=LEARNING_RATE,
#     device=DEVICE
# )
# agent.load_model("my_agent.pth") # Optional: to continue training

print("Starting training...")

# --- TODO: Write the main training loop ---
# This is just one example of an on-policy (like PPO) training loop.
# An off-policy loop (like DDPG/SAC) would look different.

obs, info = env.reset()
global_step = 0

while global_step < TOTAL_TIMESTEPS:
    
    # --- 1. Collect a batch of experiences ---
    # (You will need to create lists or buffers to store these)
    # trajectory_buffer = [] 
    
    print(f"Collecting trajectory... (Step {global_step}/{TOTAL_TIMESTEPS})")
    
    for _ in range(STEPS_PER_COLLECT):
        # --- TODO: Get an action from your agent's policy ---
        # action, log_prob, value = agent.get_action_and_value(obs)
        action = env.action_space.sample() # Placeholder: Replace with your agent's action
        
        # --- TODO: Step the environment ---
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # --- TODO: Store the transition in your buffer ---
        # e.g., trajectory_buffer.append( (obs, action, log_prob, reward, terminated, value) )

        global_step += 1
        obs = next_obs
        
        # Handle episode end
        if terminated or truncated:
            print(f"Episode finished at step {global_step}.")
            obs, info = env.reset()

    # --- 2. Update the agent's policy ---
    # (This is where you'd calculate advantages, PPO clip loss, etc.)
    print("Updating policy...")
    # --- TODO: Call your agent's update/learn function ---
    # agent.learn(trajectory_buffer)

    # --- 3. Save the model periodically ---
    if global_step % 50000 == 0:
        save_path = f"my_agent_logs/model_step_{global_step}.pth"
        # --- TODO: Implement your agent's save method ---
        # agent.save_model(save_path)
        print(f"Model saved to {save_path}")


# --- TODO: Final save and cleanup ---
# agent.save_model("my_agent_final.pth")
env.close()
print("Training finished.")