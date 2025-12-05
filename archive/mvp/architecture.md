

## Action Space
- What the environment expects: 16 continuous numbers for the joints
- What the discrete translator will provide:
    - Action 0: GRASP
        -close all fingers together
        np.full(16, -0.03) #all 16 joints move to -0.03
    - Action 1: Rotate
        - coordinated push pattern to spin cube
        - initialize: np.zeros(16)
        - joint_positions[0:8] = 0.03 -> first 8 joints move one way
        - joint_positions[8:16] = -0.03 -> last 8 joints push the other way


## observation space
- the 7 numbers rleated to the cube
- rotation around z-axis
    - z_rotation_angle = quaternion_to_z_angle(quartnerion)
- 2 features
    - cube distance from palm: captures height relative to palm
    - angular velocity: captures rotation pspeed

# archiecture

*translator.py* - this will translate the continuous state values into managable discrete states
    -bins: will sort the continuous actions
    - states:
        - cube rotating
        - cube falling
        - cube held tight
    - actions:
        - grasp
        - rotate

    -the base environment is CanRotateEnv()
        -this needs to be wrapped in the translator
    -before translation
        env.action_space = Box(shape=16, low=-0.03, high=0.03) -> 16 continuous numbers
        observation_space = env.reset() -> 23 continuous floats

    -after translation
        ACTION:
            env = translator(env)
            env.action_space = Discrete(2)
        STATE:
            obs = env.reset()
            discrete_state = make_discrete()
            state_id = discrete_state.make_discrete(obs)



*q_agent.py* - implements the q-table
    - q-table storage (numpy array)
    - epsilon-greedy action selection
    - q-value update rule
    - methods: action(), update()

*train.py* - training loop
    - uses the translator to translate the continuous actions into discrete ones
    - run episodes
    - update() table after each step
    - track rewards and save best table
    - plots learning progress

