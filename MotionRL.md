# MotionRL: Q-Learning for Robot Navigation and Obstacle Avoidance

## Table of Contents
1. [Overview](#overview)
2. [Environment Setup](#environment-setup)
3. [Q-Learning Algorithm](#q-learning-algorithm)
4. [Implementation Details](#implementation-details)
5. [Training Configuration](#training-configuration)
6. [Usage Instructions](#usage-instructions)
7. [Troubleshooting](#troubleshooting)

## Overview

This project implements Q-Learning for training a MiR100 robot to navigate from a starting point to a destination while avoiding obstacles in a simulated Gazebo environment using the robo-gym framework.

### Key Features
- **Discrete Q-Learning** for continuous control problem
- **State discretization** for position, distance, and laser sensor data
- **Epsilon-greedy exploration** with decay
- **Headless training** for faster convergence
- **Persistent Q-table** storage and loading
- **Real-time navigation** with trained policy

## Environment Setup

### Prerequisites
- **Ubuntu 20.04** with ROS Noetic
- **Python 3.8+**
- **robo-gym** framework
- **Gazebo** simulation environment

### Dependencies Installation

```bash
# Install required Python packages
pip install gymnasium numpy matplotlib

# Install robo-gym (follow official installation guide)
git clone https://github.com/jr-robotics/robo-gym.git
cd robo-gym
pip install -e .
```

### Environment Configuration

The training uses the `ObstacleAvoidanceMir100Sim-v0` environment which provides:

- **Robot**: MiR100 mobile robot
- **World**: Lab environment (6x8m) with obstacles
- **Sensors**: 16-beam laser scanner
- **Action Space**: Continuous [linear_velocity, angular_velocity]
- **Observation Space**: Robot pose, target pose, laser readings

### ROS Environment Setup

When training is running, robo-gym creates its own ROS master. To access ROS topics:

```bash
# Find the ROS master port
ps aux | grep rosmaster

# Set environment variables (replace PORT with actual port)
export ROS_MASTER_URI=http://localhost:PORT

# Now you can use ROS commands
rostopic list
rostopic echo /scan
```

## Q-Learning Algorithm

### Algorithm Overview

The implemented Q-Learning algorithm follows the standard temporal difference learning approach adapted for continuous state-action spaces through discretization.

### Pseudocode

```
procedure INITIALIZE_Q_LEARNING_AGENT(action_space, lr, gamma, epsilon)
    discrete_actions ← CREATE_DISCRETE_ACTIONS()
    Q_table ← empty dictionary with default zeros
    learning_rate ← lr
    discount_factor ← gamma
    epsilon ← epsilon
    return agent
end procedure

procedure CREATE_DISCRETE_ACTIONS()
    actions ← empty list
    linear_speeds ← [-1.0, -0.5, 0.0, 0.5, 1.0]
    angular_speeds ← [-1.0, -0.5, 0.0, 0.5, 1.0]
    for lin_vel in linear_speeds do
        for ang_vel in angular_speeds do
            actions.append([lin_vel, ang_vel])
    return actions
end procedure

procedure DISCRETIZE_STATE(state)
    robot_x ← state[0]
    robot_y ← state[1]
    target_x ← state[2] 
    target_y ← state[3]
    laser_data ← state[-16:]
    
    rel_x ← target_x - robot_x
    rel_y ← target_y - robot_y
    distance ← sqrt(rel_x² + rel_y²)
    
    rel_x_bin ← discretize(rel_x, POSITION_BINS)
    rel_y_bin ← discretize(rel_y, POSITION_BINS)
    distance_bin ← discretize(distance, POSITION_BINS)
    min_laser_bin ← discretize(min(laser_data), LASER_BINS)
    
    return (rel_x_bin, rel_y_bin, distance_bin, min_laser_bin)
end procedure

procedure CHOOSE_ACTION(state, Q_table, epsilon)
    discrete_state ← DISCRETIZE_STATE(state)
    if random() < epsilon then
        action_idx ← random_choice(num_actions)
    else
        action_idx ← argmax Q_table[discrete_state]
    return action_idx, discrete_actions[action_idx]
end procedure

procedure UPDATE_Q_TABLE(state, action_idx, reward, next_state, done, Q_table, lr, gamma)
    discrete_state ← DISCRETIZE_STATE(state)
    discrete_next_state ← DISCRETIZE_STATE(next_state)
    
    current_q ← Q_table[discrete_state][action_idx]
    
    if done then
        target_q ← reward
    else
        next_max_q ← max(Q_table[discrete_next_state])
        target_q ← reward + gamma × next_max_q
    
    Q_table[discrete_state][action_idx] ← current_q + lr × (target_q - current_q)
end procedure

procedure TRAIN_Q_LEARNING(num_episodes)
    agent ← INITIALIZE_Q_LEARNING_AGENT(action_space, lr, gamma, epsilon)
    agent.LOAD_Q_TABLE('q_table_mir100.pkl')
    
    for episode ← 1 to num_episodes do
        state ← env.reset(start_pose=START_POINT, target_pose=DESTINATION_POINT)
        done ← false
        
        while not done do
            action_idx, action ← CHOOSE_ACTION(state, agent.Q_table, agent.epsilon)
            next_state, reward, terminated, truncated, info ← env.step(action)
            done ← terminated or truncated
            
            UPDATE_Q_TABLE(state, action_idx, reward, next_state, done, 
                          agent.Q_table, agent.learning_rate, agent.discount_factor)
            
            state ← next_state
        
        agent.epsilon ← max(agent.epsilon_min, agent.epsilon × agent.epsilon_decay)
        
        if episode mod 100 = 0 then
            PRINT_PROGRESS(episode, rewards, success_rate)
    
    agent.SAVE_Q_TABLE('q_table_mir100.pkl')
end procedure

procedure NAVIGATE_WITH_TRAINED_POLICY(Q_table, start_point, destination_point)
    state ← env.reset(start_pose=start_point, target_pose=destination_point)
    done ← false
    
    while not done do
        discrete_state ← DISCRETIZE_STATE(state)
        action_idx ← argmax Q_table[discrete_state]  // Pure exploitation
        action ← discrete_actions[action_idx]
        state, reward, terminated, truncated, info ← env.step(action)
        done ← terminated or truncated
    
    return info['final_status']
end procedure
```

### Reward Structure

The environment provides rewards based on:

- **Distance to target**: -50 × euclidean_distance (encourages moving toward goal)
- **Power consumption**: -|linear_velocity| × 0.30 - |angular_velocity| × 0.03
- **Success**: +100 (reaching target within 0.2m threshold)
- **Collision**: -200 (hitting obstacles or laser threshold violation)

### Termination Conditions

- **terminated = True**: 
  - Robot reaches target (success)
  - Robot collides with obstacle (failure)
  - Minimum laser reading below threshold (too close to obstacle)

- **truncated = True**:
  - Maximum episode steps exceeded (500 steps)

## Implementation Details

### State Discretization Parameters

```python
POSITION_BINS = 20      # Discretization for x, y coordinates
LASER_BINS = 5          # Discretization for laser readings
MAX_DISTANCE = 5.0      # Maximum distance for normalization
```

### Q-Learning Hyperparameters

```python
LEARNING_RATE = 0.1     # Step size for Q-value updates
DISCOUNT_FACTOR = 0.95  # Future reward discount factor
EPSILON = 1.0           # Initial exploration rate
EPSILON_DECAY = 0.995   # Exploration decay per episode
EPSILON_MIN = 0.01      # Minimum exploration rate
NUM_EPISODES = 1000     # Training episodes
```

### Action Space Discretization

The continuous action space [linear_velocity, angular_velocity] is discretized into 25 discrete actions:
- Linear velocities: [-1.0, -0.5, 0.0, 0.5, 1.0]
- Angular velocities: [-1.0, -0.5, 0.0, 0.5, 1.0]

## Training Configuration

### Configurable Parameters

```python
# Navigation points
START_POINT = [0.5, 0.0, 0.0]        # [x, y, yaw] starting position
DESTINATION_POINT = [2.5, 2.5, 0.0]  # [x, y, yaw] target position

# Training mode
HEADLESS_TRAINING = True              # No GUI during training
SHOW_GUI_FOR_TESTING = True           # Show GUI for testing
```

### Performance Metrics

The training tracks:
- **Episode rewards** (total reward per episode)
- **Episode lengths** (steps to completion/termination)
- **Success rate** (percentage of episodes reaching target)
- **Exploration rate** (current epsilon value)

## Usage Instructions

### Training the Agent

```bash
# Run training (headless mode for speed)
python testing.py
```

Training output:
```
Training Configuration:
  Headless Mode: True
  Episodes: 1000

Episode 100/1000
  Average Reward (last 100): -45.23
  Average Length (last 100): 234.5
  Success Rate: 15.20%
  Epsilon: 0.366

✓ Q-table saved successfully to q_table_mir100.pkl
```

### Running Navigation with Trained Agent

```bash
# Navigate using trained policy
python robot_avoidance_ql.py
```

Navigation output:
```
🚀 Starting navigation...
   From: [0.5, 0.0, 0.0]
   To: [2.5, 2.5, 0.0]

📍 Navigation in progress...
   Step 50: Distance to target = 2.1m

📊 Navigation completed!
   🎉 SUCCESS! Robot reached the destination!
   Total Steps: 127
   Total Reward: 45.30
```

### File Structure

```
robo-gym/
├── testing.py              # Q-Learning training script
├── robot_avoidance_ql.py   # Navigation with trained agent
├── q_table_mir100.pkl      # Saved Q-table (generated)
├── training_results.png    # Training plots (generated)
└── MotionRL.md            # This documentation
```

## Troubleshooting

### Common Issues

1. **ROS Master Connection Error**
   ```
   ERROR: Unable to communicate with master!
   ```
   **Solution**: Set correct ROS master URI
   ```bash
   export ROS_MASTER_URI=http://localhost:57639  # Use actual port
   ```

2. **Environment Creation Fails**
   - Ensure robo-gym server modules are properly installed
   - Check if required ROS packages are available
   - Verify Gazebo can start properly

3. **Training Convergence Issues**
   - Increase number of episodes
   - Adjust learning rate or exploration parameters
   - Check reward function alignment with task goals

4. **Low Success Rate**
   - Verify start and destination points are reachable
   - Check obstacle configuration in environment
   - Consider adjusting discretization parameters

### Performance Optimization

- **Headless training**: Set `HEADLESS_TRAINING = True` for 2-5x speedup
- **Parallel training**: Multiple environment instances (advanced)
- **Hyperparameter tuning**: Adjust learning rate and exploration schedule

### Monitoring Training Progress

1. **Real-time metrics**: Check console output every 100 episodes
2. **ROS topics**: Monitor sensor data and robot state
   ```bash
   rostopic echo /scan
   rostopic echo /odom
   ```
3. **Visualization**: Use training plots to analyze convergence

---

*This documentation covers the Q-Learning implementation for robot navigation and obstacle avoidance using the robo-gym framework. For additional details, refer to the source code comments and robo-gym official documentation.*