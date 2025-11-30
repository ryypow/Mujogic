"""
This translator will convert 2 discrete actions into the 16 individual movements for all 16 joints
based on the OpenAI Gym wrapper tutorial
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ActionTranslator(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        #define the discrete action spaces
        self.action_space = spaces.Discrete(3) #two action features - GRASP and ROTATE

    def action(self, act):
        """
        Convert discrete action to continuous actions
        two discrete actions:
            - action 0 = grasp, which closes all fingers
            - action 1 = rotate, which makes half of the joints close and half open
            - action 2 = hold, keeps jpoints in current position
        """
        continuous = np.zeros(16)

        if act == 0: #strong grasp - close all
            continuous[:] = 0.03
        
        elif act == 1: #strong release - open all
            continuous[:] = -0.03

        elif act == 2: #rotation - testing finger2 as the anchor and 
            continuous[0:4] = -0.03 #finger1 opens
            continuous[4:8] = 0.00 #finger2 is anchored, no movement from current pos
            continuous[8:12] = 0.03 #finger3 closes and should push the cube
            continuous[12:16] = 0.01 #thumb closes slightly

        elif act == 3: #hold at target
            continuous[:] = 0.0 #this will keep the fingers in their current position, stopping rotation
            #NOTE: this may not work due to the momentum of the cubes rotation

        #elif act == 3: #close all fingers until contact


        return continuous
    
    # Add this method to ActionTranslator for debugging
    def test_action(self, env, action_id, steps=50):
        """Visualize what an action does"""
        env.reset()
        for _ in range(steps):
            obs, reward, _, _, _ = env.step(action_id)
            env.render()
        print(f"{self.action_names[action_id]}: Cube rotated {np.rad2deg(obs[-4:])} degrees")
    
    
