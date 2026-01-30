# inhand_test.py - Updated for DQN

import time
import torch
import numpy as np
from inhand_env import CanRotateEnv
from MinimalTranslator import MinimalTranslator
import DQNagent

# --- Configuration ---
MODEL_PATH = "dqn_final.pth" #"dqn_episode_600.pth"
EPISODES_TO_RUN = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load Environment (with wrapper) ---
env = CanRotateEnv(render_mode="human")
env = MinimalTranslator(env)

# --- Load trained DQN ---
NUM_STATES = env.observation_space.shape[0]
NUM_ACTIONS = env.action_space.n

policy_net = DQNagent.Net(obs_space=NUM_STATES, actions=NUM_ACTIONS).to(device)
policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
policy_net.eval()  # Set to evaluation mode

print(f"Loaded model from {MODEL_PATH}")

# --- Run Evaluation ---
for episode in range(EPISODES_TO_RUN):
    print(f"--- Starting Episode {episode + 1} ---")
    obs, info = env.reset()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    while not (terminated or truncated):
        # Deterministic action (no exploration)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            action = q_values.argmax(dim=1).item()
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        time.sleep(1/60)
        
    print(f"Episode {episode + 1} finished. Total Reward: {total_reward:.2f}")

env.close()
print("\nEvaluation finished.")
