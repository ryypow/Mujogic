import numpy as np

  # Load old Q-table (72 states)
old_q_table = np.load('q_table.npy')
print(f"Old shape: {old_q_table.shape}")  # (72, 5)

  # Extract only the +Z direction states (0-35)
new_q_table = old_q_table[:36, :]
print(f"New shape: {new_q_table.shape}")  # (36, 5)

  # Save the converted Q-table
np.save('q_table_36states.npy', new_q_table)
print("Saved converted Q-table")

  # Verify
print(f"Non-zero in old: {np.count_nonzero(old_q_table)}")
print(f"Non-zero in new: {np.count_nonzero(new_q_table)}")
