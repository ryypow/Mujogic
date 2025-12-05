# inhand_test.py (Student Skeleton)
import time
from inhand_env import CanRotateEnv

# --- TODO: Import your agent class ---
# from agent import MyRLAgent 

# --- Configuration ---
MODEL_PATH = "my_agent_final.pth" # Path to your saved student model
EPISODES_TO_RUN = 10

# --- TODO: Load the environment ---
env = CanRotateEnv(render_mode="human")

# --- TODO: Load your trained agent ---
# agent = MyRLAgent(
#     obs_space_shape=env.observation_space.shape,
#     action_space_shape=env.action_space.shape,
#     device='cpu'
# )
# try:
#     agent.load_model(MODEL_PATH)
#     print(f"Successfully loaded model from {MODEL_PATH}")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     exit()


# --- Run the evaluation ---
for episode in range(EPISODES_TO_RUN):
    print(f"--- Starting Episode {episode + 1} ---")
    
    # --- TODO: Reset the environment ---
    obs, info = env.reset()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    while not (terminated or truncated):
        
        # --- TODO: Get a deterministic action from your agent ---
        # The 'deterministic=True' part is key for testing
        # action = agent.get_action(obs, deterministic=True)
        action = env.action_space.sample() # Placeholder: Replace with your agent's action
        
        # --- TODO: Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        
        # Render/sleep is handled by the environment's step/render methods
        time.sleep(1/60) # Keep visualization smooth
        
    print(f"Episode {episode + 1} finished. Total Reward: {total_reward:.2f}")

# Clean up
env.close()
print("\nEvaluation finished.")