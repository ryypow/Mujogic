Q-TABLE VERSION 3: 288 States x 5 Actions (with Goal and Rotation Bins)
========================================================================


STATE SPACE (288 states)
------------------------
State ID = rotation_bin * 144 + goal_bin * 36 + grasp_bin * 12 + speed_bin * 4 + progress_bin

Components:
1. ROTATION_BIN (2 bins):
   - 0 = current z_rotation >= 0 (positive side)
   - 1 = current z_rotation < 0 (negative side)

2. GOAL_BIN (4 bins):
   - 0 = small goal (0-45 degrees)
   - 1 = medium goal (46-90 degrees)
   - 2 = large goal (91-180 degrees)
   - 3 = very large goal (181-360 degrees)

3. GRASP_BIN (3 bins):
   - 0 = weak grasp (0-1 fingers)
   - 1 = stable grasp (2 fingers)
   - 2 = strong grasp (3+ fingers)

4. SPEED_BIN (3 bins):
   - 0 = not spinning (< 0.02 rad/s)
   - 1 = rotating (0.02-0.1 rad/s)
   - 2 = fast rotation (>= 0.1 rad/s)

5. PROGRESS_BIN (4 bins - RELATIVE):
   - 0 = far (> 70% of goal remaining)
   - 1 = making progress (30-70% remaining)
   - 2 = close (10-30% remaining)
   - 3 = at goal (< 10% remaining)

Total: 2 x 4 x 3 x 3 x 4 = 288 states

ACTION SPACE (5 actions)
------------------------
0 = GRASP    - Close all fingers
1 = RELEASE  - Open all fingers
2 = ROT_POS  - Rotate positive (+Z direction)
3 = HOLD     - Keep position
4 = ROT_NEG  - Rotate negative (-Z direction)

HYPERPARAMETERS
---------------
LEARNING_RATE = 0.3 (higher to preserve knowledge)
DISCOUNT = 0.99
EPSILON = 0.6-0.9 (varied)
EPSILON_DECAY = 0.998
MIN_EPSILON = 0.05
NUM_EPISODES = 1500-2000
MAX_STEPS = 400
GOAL = [30, 45, 60, 90] degrees (curriculum)

KEY FEATURES
------------
- RELATIVE progress bins (scales to any goal size)
- Goal magnitude bin allows learning different strategies per goal
- Rotation bin tracks current rotation position
- Curriculum learning: randomizes goals from list each episode
- Checkpoint saves every 100 episodes

FILES (sorted by episode)
-------------------------
Q-Tables:
  q_table_checkpoint_100.npy  - Episode 100 checkpoint
  q_table_checkpoint_200.npy  - Episode 200 checkpoint
  ...
  q_table_checkpoint_1500.npy - Episode 1500 checkpoint
  q_table_checkpoint_1900.npy - From training_288states folder
  q_table_checkpoint_2000.npy - From training_288states folder
  q_table_final.npy           - Final from mvp/ (150 non-zero, max=157.65)
  q_table_final1.npy          - Alternative final (19 non-zero, max=8.20)

Training Logs:
  training_debug_output.txt - Large debug log (~488KB) showing per-step distance
                             tracking during training. Shows "current distance"
                             for each timestep, useful for analyzing convergence
                             and oscillation patterns during 288-state training.

TRAINING SESSIONS
-----------------
Session 1 (mvp/q_agent_finetune.py):
  - GOAL = [45] only
  - 1500 episodes
  - Checkpoints: 100-1500

Session 2 (training_288states/q_agent_finetune.py):
  - GOAL = [30, 60, 90]
  - 2000 episodes
  - Checkpoints: 1900, 2000

COMMIT HISTORY
--------------
6d86a1d - 288 states - added direction_bin (misnamed, actually rotation_bin)
3830df9 - Training with rotation_bin
4f49c9c - Changed direction_bin to rotation_bin, adjusted hyperparameters
3b869ae - Q-tables generated for first 1000 episodes
f70c67d - Increased state space to prevent forgetting

NOTE
----
This version was designed to handle multiple goal sizes with a single
Q-table by using relative progress bins. The rotation_bin helps the
agent understand its current position in the rotation space.
