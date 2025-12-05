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

        if act == 0: #grasp
            continuous[:] = -0.02

        elif act == 1: #rotate - half open and half close
            continuous[0:4] = -0.02 #finger 1 and 2 extends
            continuous[4:8] = 0.01 #finger 2 closes slightly
            continuous[8:12] = 0.02 #finger 3  closes 
            continuous[12:16] = 0.00 #thumb anchors

        elif act == 2: #hold at target
            continuous[:] = 0.0 #this will keep the fingers in their current position, stopping rotation
            #NOTE: this may not work due to the momentum of the cubes rotation

        return continuous
    

    
import gymnasium
import numpy as np
from gymnasium import spaces

class ActionTranslator(gymnasium.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        d = 16                      # number of joints in your hand
        strong = 0.03               # big step
        fine   = 0.01               # small step

        # convenience vectors
        ones  = np.ones(d)
        half  = np.ones(d // 2)

        rotate_cw  = np.concatenate([ strong * half, -strong * half])
        rotate_ccw = np.concatenate([-strong * half,  strong * half])

        self.action_table = np.array([
            -strong * ones,         # 0: strong close
            +strong * ones,         # 1: strong open
            -fine   * ones,         # 2: fine close
            +fine   * ones,         # 3: fine open
            rotate_cw,              # 4: rotate CW
            rotate_ccw,             # 5: rotate CCW
            np.zeros(d),            # 6: hold / no-op
        ], dtype=np.float32)

        self.action_space = spaces.Discrete(len(self.action_table))

    def action(self, act: int):
        return self.action_table[act].copy()
