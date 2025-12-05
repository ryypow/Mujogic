# agent.py
import numpy as np
import os
import torch
import torch.nn as nn
from Translator import ActionTranslator, ObsTranslator

class MyRLAgent:
    def __init__(self, obs_space_shape, action_space_shape, learning_rate=3e-4, device='cpu'):
        self.obs_space = obs_space_shape
        self.action_space = action_space_shape
        self.learning_rate = learning_rate

        
    def get_action_and_value(self, obs):
        if np.random.uniform(0,1) < EPSILON:
            action = ActionTranslator.action.sample()
        else:
            action = np.argmax(q_table[state])
    
    def learn(self, trajectory_buffer):

    
    def save_model(self, path):
        np.save('q_table_v2_final.npy', q_table)
    
    def load_model(self, path):
        Q_TABLE_PATH = 'q_table_v2.npy'

        if os.path.exists(Q_TABLE_PATH):
            print(f"Loading existing Q-table from {Q_TABLE_PATH}")
            q_table = np.load(Q_TABLE_PATH)
            if q_table.shape != (NUM_STATES, NUM_ACTIONS):
                print(f"Shape mismatch! Reinitializing...")
                q_table = np.zeros((NUM_STATES, NUM_ACTIONS))
        else:
            print("Initializing new Q-table with zeros")
            q_table = np.zeros((NUM_STATES, NUM_ACTIONS))

        print(f"Q-table shape: {q_table.shape}")


        
        