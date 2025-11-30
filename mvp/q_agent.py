"""
Tabular Q-learning for cube rotation -> rotate to exact angles and stop
based on: https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
"""
import sys
import os
from discrete_translator import ActionTranslator
#env_path = os.path.abspath('./env')
#sys.path.append(env_path)

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from PIL import Image
from inhand_env import CanRotateEnv


def state_translator(observation, env):

    base_env = env.unwrapped #Gym stores the unwrapped environment in .unwrapped -> needed for palm position


    # ==== discrete bin for progress towards goal
    cube_quart = observation[-4:] #quartnerion orientatoin for cube
    r = Rotation.from_quat(cube_quart) #reminder: working with radians
    euler = r.as_euler('xyz') #x,y,z coordinates
    z_rotation = euler[2]    

    goal = base_env.target_rotation
    goal_progress = abs(goal - z_rotation)

    if goal_progress > np.deg2rad(30):
        progress_bin = 0 #far from goal
    elif goal_progress > np.deg2rad(15):
        progress_bin = 1 #getting there
    elif goal_progress > np.deg2rad(5): #agreeing with configured tolerance
        progress_bin = 2 #acceptable
    else:
        progress_bin = 3 #winner winner chicken dinner

    #testing grasp strength - fingers in contact (from reward function)
    fingers_in_contact = set()  # the reward function uses a set
    
    for i in range(base_env.sim.data.ncon):
        contact = base_env.sim.data.contact[i]
        geom1, geom2 = contact.geom1, contact.geom2
        
        if geom1 in base_env.fingertip_geom_ids and geom2 == base_env.can_geom_id:
            fingers_in_contact.add(geom1)
        elif geom2 in base_env.fingertip_geom_ids and geom1 == base_env.can_geom_id:
            fingers_in_contact.add(geom2)
    
    num_fingers = len(fingers_in_contact)
    
    if num_fingers <= 1:
        grasp_bin = 0  # weak
    elif num_fingers == 2:
        grasp_bin = 1  # stable
    else:  # >= 3
        grasp_bin = 2  # strong


    #==== discrete bin for speed
    #rotation velocity - from env.calculate_reward()
    obj_vel = np.zeros(6)
    mujoco.mj_objectVelocity(base_env.sim.model, base_env.sim.data, mujoco.mjtObj.mjOBJ_BODY, base_env.obj_body_id, obj_vel, 0)
    angular_velocity_z = abs(obj_vel[2]) #spin speed

    #rotation speed bins - unsure about actual speed, adjust accordingly TODO.txt
    if angular_velocity_z < 0.1:
        speed_bin = 0 #not spinning
    elif angular_velocity_z < 0.5:
        speed_bin = 1 #rotating
    else:
        speed_bin = 2 #rotating fast

    """   
    #the state features are combined to form the state ID's
    state_map = {
         (0,0): 0, #the cube is close to the palm and not spinning
         (0,1): 1, #the cube is close to the palm and spinning slowly
         (0,2): 2, #the cube is close to the palm and spinning fastly
         (1,0): 3,#the cube is medium distance from the palm and not spinning
         (1,1): 4,#the cube is medium distance from the palm and spinning slowly
         (1,2):5,#the cube is medium distance from the palm and spinning fastly
         (2,0): 6,#the cube is far from the palm and not spinning
         (2,1): 7,#the cube is far from the palm and spinning slowly
         (2,2): 8#the cube is far from the palm and spinning fastly
     }

    state_id = state_map[(dist_bin, speed_bin)]
    """

    #changed from state map to 3D index
    #progress bin has 4 values
    #dist bin has 3 values
    #speed bin has 3 values
    #dist bin multiplied by the speed_bin*progress_bin
    #speed bin is multiplied by the amount of progress_bin options
    state_id = grasp_bin * 12 + speed_bin * 4 + progress_bin

    return state_id

#============================
# training parameters
#===========================

NUM_STATES = 36 #3 dist_bin's * 3 speed_bin's * 4 progress_bin's
NUM_ACTIONS = 3 #Action 0 = grasp, action 1 = rotate, action 2 = hold (0.0 -> current position)
LEARNING_RATE = 0.1 #ALPHA -> how fast to update q-values
DISCOUNT = 0.95 #GAMMA -> future reward importance
EPSILON = 1.0 #high epsilon = 100% exploration rate
EPSILON_DECAY = 0.999 #the rate at which exploration will be reduced, prioritizing exploitation
MIN_EPSILON = 0.01 #Always explore at least 1%
NUM_EPISODES = 1000 #EPISODES TO TRAIN
MAX_STEPS = 300
DEVICE = 'cpu'
GOAL = 90 #[90,180,270,360]

#==========================
# INITIALIZE THE Q-TABLE: 9 rows/states, 2 columns/actions
#==========================
print("Initializing q-table")
q_table = np.zeros((NUM_STATES, NUM_ACTIONS))
print("\nq-table initialized with zeros")
print("\nQ-table shape: ", q_table.shape)

#==========================
# initialize environment and the discrete translator
#==========================

base_env = CanRotateEnv(GOAL, render_mode="headless")
env = ActionTranslator(base_env)

#==========================
# training loop
#==========================
print("Starting training...")
os.makedirs("termination_snap", exist_ok=True) #collect images of termination
reward_tracker = []

for episode in range(NUM_EPISODES):
    observation,_ = env.reset() #reset env for before each episode begins
    state = state_translator(observation, env) #sends observation space to the translator to get discrete bins
    step = 0
    total_reward = 0.0
    done = False

    while True:
        if np.random.uniform(0,1) < EPSILON: #agent will prefer exploration initially, until the epsilon decays
            action = env.action_space.sample() #returns 0 for grasp or 1 for rotate
        else:
            action = np.argmax(q_table[state]) #returns action from q-table

        #execute action
        #next_observation is an array containing the new joint positions, object position, and object orientation
        next_observation, reward, terminated, truncated, _ = env.step(action)
        

        """
        #Get cubes orientation, get euler angle via scipy.Rotation
        cube_quart = next_observation[-4:]
        r = Rotation.from_quat(cube_quart) #reminder: working with radians
        euler = r.as_euler('xyz')
        z_rotation = euler[2]
        """

        #escape if terminated/truncated
        if terminated or truncated:
            #frame = env.render()
            #img = Image.fromarray(frame)
            #img.save(f"termination_snap/episode_{episode}.png")
            break

        #translate new positional values into the discrete bins
        new_state = state_translator(next_observation, base_env)

        #old q-score for state, action
        prev_q = q_table[state, action]

        #best possible score for the new state
        best_future_q = np.max(q_table[new_state])

        #calculate new q
        #old_score + immediate reward + (discount * best_future_q -> best possible future reward)
        #learning rate adjusts the error slightly
        #where the error is the discounted reward multiplied by the difference between best_q and previous_q
        new_q = prev_q + LEARNING_RATE * (reward + DISCOUNT * best_future_q - prev_q)

        q_table[state,action] = new_q

        state = new_state
        total_reward += reward
    
    reward_tracker.append(total_reward)

    #adjust policy
    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

    if (episode + 1) % 10 == 0:
        average_reward = np.mean(reward_tracker[-50:])
        print(f"Episode {episode+1}/{NUM_EPISODES} - Avg Reward: {average_reward:.2f} - Epsilon: {EPSILON:.3f} - Steps: {step}")

np.save('q_table.npy', q_table)
print("model saved as q_table.npy")
