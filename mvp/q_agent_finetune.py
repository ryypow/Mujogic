"Fine-tuning q_table for larger goals and reverse rotation (favors positive right now)"

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
    #MuJoCo quaternion format: [w, x, y, z], SciPy expects: [x, y, z, w]
    cube_quat_mujoco = observation[-4:]  # [qw, qx, qy, qz]
    cube_quat_scipy = np.array([cube_quat_mujoco[1], cube_quat_mujoco[2], cube_quat_mujoco[3], cube_quat_mujoco[0]])
    r = Rotation.from_quat(cube_quat_scipy)
    euler = r.as_euler('xyz') #x,y,z coordinates
    z_rotation = euler[2]    

    goal = base_env.target_rotation
    goal_progress = goal - z_rotation

    #new
    #irection_bin = 0 if goal_progress >= 0 else 1 #0 = need positive rotation, 1 = need negative rotation
    #goal_progress_abs = abs(goal_progress)

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

        if geom1 in base_env.fingertip_geom_ids and geom2 in base_env.can_geom_ids:
            fingers_in_contact.add(geom1)
        elif geom2 in base_env.fingertip_geom_ids and geom1 in base_env.can_geom_ids:
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
    if angular_velocity_z < 0.02:
        speed_bin = 0 #not spinning
    elif angular_velocity_z < 0.1:
        speed_bin = 1 #rotating
    else:
        speed_bin = 2 #rotating fast

    state_id = grasp_bin * 12 + speed_bin * 4 + progress_bin

    return state_id, progress_bin, grasp_bin, speed_bin, z_rotation, goal_progress

#============================
# training parameters
#===========================

NUM_STATES = 36 #3 grasp, 3 speed, 4 progress
NUM_ACTIONS = 5
LEARNING_RATE = 0.05 #lowered from 0.1 -- need to preserve existing knowledge
DISCOUNT = 0.99 #GAMMA -> long-horizong importance
EPSILON = 0.3 #decreased, since we are using a policy
EPSILON_DECAY = 0.997 #increased from 0.995 to 0.997 - prioritizing exploioration
MIN_EPSILON = 0.05 #increased to 5% - maintain exploration
NUM_EPISODES = 4000 #EPISODES TO TRAIN - 1000 for each goal
MAX_STEPS = 500 #increase steps from 300 to 500
DEVICE = 'cpu'
GOAL = [90, 180, 270, 360] #Start small, increase once agent learns (30->45->60->90)
ACTION_NAMES = {
    0: "GRASP",
    1: "RELEASE",
    2: "ROT_POS",   # Rotate toward +Z (positive direction)
    3: "HOLD",
    4: "ROT_NEG"    # Rotate toward -Z (negative direction)
}

#==========================
# INITIALIZE THE Q-TABLE: 9 rows/states, 2 columns/actions
#==========================
import os

Q_TABLE_PATH = 'q_table.npy'

if os.path.exists(Q_TABLE_PATH):
    print(f"Loading existing Q-table from {Q_TABLE_PATH}")
    q_table = np.load(Q_TABLE_PATH)
    print(f"Loaded Q-table shape: {q_table.shape}")

    # Verify shape matches current state/action space
    if q_table.shape != (NUM_STATES, NUM_ACTIONS):
        print(f"Warning: Shape mismatch! Expected ({NUM_STATES}, {NUM_ACTIONS})")
        print("Initializing new Q-table with zeros")
        q_table = np.zeros((NUM_STATES, NUM_ACTIONS))
else:
    print("No existing Q-table found. Initializing with zeros")
    q_table = np.zeros((NUM_STATES, NUM_ACTIONS))

print(f"Q-table shape: {q_table.shape}")


#==========================
# training loop
#==========================
print("Starting training...")
os.makedirs("termination_snap", exist_ok=True) #collect images of termination
reward_tracker = []

for episode in range(NUM_EPISODES):

    #randomizing the goal for each episode
    goal = np.random.choice(GOAL)
    base_env = CanRotateEnv(goal, render_mode="headless")
    env = ActionTranslator(base_env)


    observation,_ = env.reset() #reset env for before each episode begins
    state, progress_bin, grasp_bin, speed_bin, z_rot, goal_prog = state_translator(observation, env)
    step = 0
    total_reward = 0.0
    done = False
    base = env.unwrapped

    while step < MAX_STEPS:
        if np.random.uniform(0,1) < EPSILON: #agent will prefer exploration initially, until the epsilon decays
            action = env.action_space.sample() #returns 0 for grasp or 1 for rotate
        else:
            action = np.argmax(q_table[state]) #returns action from q-table

        #execute action
        #next_observation is an array containing the new joint positions, object position, and object orientation
        next_observation, reward, terminated, truncated, _ = env.step(action)

        new_state, progress_bin, grasp_bin, speed_bin, z_rot, goal_prog = state_translator(next_observation, env)

        total_reward += reward
        step += 1



        #old q-score for state, action

        if terminated:
            if goal_prog < np.deg2rad(5):
                best_future_q = np.max(q_table[new_state]) #goal achieved
            else:
                best_future_q = 0 #dropped cube - true failure
                
            prev_q = q_table[state, action]
            new_q = prev_q + LEARNING_RATE * (reward + DISCOUNT * best_future_q - prev_q)
            q_table[state, action] = new_q            
            break
        
        elif truncated:
            best_future_q = np.max(q_table[new_state]) #time limit reached - not a failure
        else:
            best_future_q = np.max(q_table[new_state])

        prev_q = q_table[state, action]
        new_q = prev_q + LEARNING_RATE * (reward + DISCOUNT * best_future_q - prev_q)

        q_table[state,action] = new_q

        state = new_state



    
    reward_tracker.append(total_reward)

    #adjust policy
    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

    if (episode + 1) % 100 == 0:
        average_reward = np.mean(reward_tracker[-100:])
        print(f"Episode {episode+1}/{NUM_EPISODES} | Avg Reward: {average_reward:.2f} | Epsilon: {EPSILON:.3f} | Last Goal: {goal}° | Final Z-Rot: {np.rad2deg(z_rot):.1f}°")


np.save('q_table_4goals.npy', q_table)
print("model saved as q_table4goals.npy")
