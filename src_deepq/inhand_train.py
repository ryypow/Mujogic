# inhand_train.py (Student Skeleton)
import os
import numpy as np
import time
import matplotlib as plt
import random
import torch
from torch import nn
import torch.nn.functional as F
from collections import deque
import gymnasium as gym
from gymnasium import spaces

from inhand_env import CanRotateEnv 
import DQNagent
from MinimalTranslator import MinimalTranslator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

env = CanRotateEnv(render_mode="headless")
env = MinimalTranslator(env) #wrap env actions

# --- Configuration ---
EPISODES = 1000
STEPS = 300
LEARNING_RATE = 0.001
DISCOUNT = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.997
MIN_EPSILON = 0.05
MEMORY = deque(maxlen=10000)

NUM_STATES = env.observation_space.shape[0]
NUM_ACTIONS = env.action_space.n


#NN
NETWORK_SYNC = 10 #THIS WILL SYNC POLICY AND TARGET NETWORK EVERY 10 STEPS
LEARN_DELAY = 1000 #starts training after 1000 transitions
BATCH = 64
LOSS_FN = nn.MSELoss() #predicts q-values
POLICY_DQN = DQNagent.Net(obs_space=NUM_STATES, actions=NUM_ACTIONS)
TARGET_DQN = DQNagent.Net(obs_space=NUM_STATES, actions=NUM_ACTIONS)
TARGET_DQN.load_state_dict(POLICY_DQN.state_dict())
TARGET_DQN.eval() #target NN will remain in eval mode

OPTIMIZER = torch.optim.Adam(POLICY_DQN.parameters(), lr=LEARNING_RATE)




def train_step():
    if len(MEMORY) < LEARN_DELAY:
        return  # Wait until enough experiences
    
    #Sample minibatch
    batch = random.sample(MEMORY, BATCH)
    states, actions, next_states, rewards, dones = zip(*batch)
    
    #Convert to tensors
    states = torch.FloatTensor(np.array(states))
    actions = torch.LongTensor(np.array(actions)).unsqueeze(1)
    next_states = torch.FloatTensor(np.array(next_states))
    rewards = torch.FloatTensor(np.array(rewards))
    dones = torch.FloatTensor(np.array(dones))
    
    #Compute current Q-values
    q_values = POLICY_DQN(states).gather(1, actions).squeeze()  # [64]
    
    #Compute target Q-values
    with torch.no_grad():
        next_q = TARGET_DQN(next_states).max(dim=1)[0]  # [64]
        q_targets = rewards + DISCOUNT * next_q * (1 - dones)
    
    #Compute loss and optimize
    loss = F.mse_loss(q_values, q_targets)
    
    OPTIMIZER.zero_grad()
    loss.backward()
    OPTIMIZER.step()
    
    return loss.item()

print("Starting training...")

obs, info = env.reset()
step_count = 0

for episode in range(EPISODES):
    state, info = env.reset()
    terminated = False
    truncated = False
    episode_reward = 0

    for step in range(STEPS):
        if random.random() < EPSILON:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_val = POLICY_DQN(state_tensor)
                action = q_val.argmax(dim=1).item()

        #perfrom action
        new_state, reward, terminated, truncated,_ = env.step(action)
        #if len(MEMORY) < BATCH:
         #    break
        #else:
        #     batch = random.sample(MEMORY, BATCH)
        episode_reward += reward

        done = terminated or truncated
        MEMORY.append((state,action,new_state,reward,done))

        step_count += 1

        train_step()
        state = new_state


        if terminated or truncated:
            break

    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
    
    if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {EPSILON:.3f}, Memory: {len(MEMORY)}")

    if episode % 100 == 0:
        TARGET_DQN.load_state_dict(POLICY_DQN.state_dict())
        torch.save(POLICY_DQN.state_dict(), f'dqn_episode_{episode}.pth')
        print(f"Episode {episode}: Target synced, model saved")

torch.save(POLICY_DQN.state_dict(), 'dqn_final.pth')
print("Training complete! Final model saved.")
