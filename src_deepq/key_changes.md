##Observation space
- the state is a continuous vector
- instead of the discrete state space like in q-learning: **state = _get_obs()**


## reward
- removed fingers in contact



-upgrades
UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at /pytorch/torch/csrc/utils/tensor_new.cpp:253.)
  states = torch.FloatTensor(states)           # [64, 20]
