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

The training uses the **Battery-Aware** `BatteryObstacleAvoidanceMir100Sim-v0` environment which provides:

- **Robot**: MiR100 mobile robot with battery management system
- **World**: Lab environment (6x8m) with obstacles
- **Sensors**: 16-beam laser scanner
- **Action Space**: Discrete forward-only actions [linear_velocity: 0.0-1.0, angular_velocity: -1.0 to 1.0]
- **Observation Space**: Robot pose, target pose, laser readings
- **Battery System**: Real-time energy consumption tracking (0-100%)

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

### Reward Structure (Battery-Aware)

The environment provides rewards based on:

- **Distance to target**: -50 × euclidean_distance (encourages moving toward goal)
- **Battery consumption**: -0.5 × total_battery_drain (encourages energy efficiency)
  - Linear movement drain: 0.08% per velocity unit per step
  - Angular movement drain: 0.03% per velocity unit per step
  - Idle consumption: 0.005% per step (sensors, computation)
- **Low battery penalty**: Additional penalty when battery < 20%
- **Critical battery penalty**: -10.0 when battery < 5%
- **Success**: +100 (reaching target within 0.2m threshold)
- **Efficiency bonus**: +(battery_level - 50) × 0.5 when completing with >50% battery
- **Collision**: -200 (hitting obstacles or laser threshold violation)

### Termination Conditions

- **terminated = True**: 
  - Robot reaches target (success)
  - Robot collides with obstacle (failure)
  - Minimum laser reading below threshold (too close to obstacle)
  - Battery depleted (0% battery) - NEW

- **truncated = True**:
  - Maximum episode steps exceeded (500 steps)

## Implementation Details

### State Discretization Parameters

```python
POSITION_BINS = 15      # Discretization for x, y coordinates (reduced for faster learning)
LASER_BINS = 3          # Discretization for laser readings (simplified)
MAX_DISTANCE = 5.0      # Maximum distance for normalization
```

### Q-Learning Hyperparameters (Optimized)

```python
LEARNING_RATE = 0.2     # Step size for Q-value updates (increased for faster learning)
DISCOUNT_FACTOR = 0.95  # Future reward discount factor
EPSILON = 1.0           # Initial exploration rate
EPSILON_DECAY = 0.999   # Exploration decay per episode (slower decay for more exploration)
EPSILON_MIN = 0.01      # Minimum exploration rate
NUM_EPISODES = 2000     # Training episodes (increased for better convergence)
```

### Battery Management Parameters

```python
INITIAL_BATTERY = 100.0          # Initial battery percentage
BATTERY_DRAIN_LINEAR = 0.08      # Battery drain per linear velocity unit
BATTERY_DRAIN_ANGULAR = 0.03     # Battery drain per angular velocity unit  
BATTERY_DRAIN_IDLE = 0.005       # Battery drain when idle
LOW_BATTERY_THRESHOLD = 20.0     # Low battery warning threshold
CRITICAL_BATTERY_THRESHOLD = 5.0 # Critical battery level
```

### Action Space Discretization (Optimized for Navigation)

The continuous action space [linear_velocity, angular_velocity] is discretized into 20 discrete actions:
- **Linear velocities**: [0.0, 0.2, 0.5, 1.0] (forward-only for better navigation)
- **Angular velocities**: [-1.0, -0.5, 0.0, 0.5, 1.0] (full turning range)

**Rationale**: Removed backward movement to encourage forward navigation and reduce action space complexity.

## Training Configuration

### Configurable Parameters (Optimized)

```python
# Navigation points (closer for easier initial learning)
START_POINT = [1.0, 1.0, 0.0]        # [x, y, yaw] starting position
DESTINATION_POINT = [2.0, 2.0, 0.0]  # [x, y, yaw] target position (1.41m distance)

# Training mode
HEADLESS_TRAINING = True              # No GUI during training (faster)
VERBOSE_LOGGING = False               # Reduced logging for performance  
SHOW_GUI_FOR_TESTING = True           # Show GUI for testing
Q_TABLE_PRINT_INTERVAL = 50          # Print Q-table summary frequency
```

### Performance Metrics (Enhanced with Battery Tracking)

The training tracks:
- **Episode rewards** (total reward per episode including battery penalties)
- **Episode lengths** (steps to completion/termination)
- **Success rate** (percentage of episodes reaching target)
- **Battery efficiency** (average battery remaining at completion)
- **Battery depletion rate** (episodes ending due to battery exhaustion)
- **Exploration rate** (current epsilon value)
- **Q-table statistics** (states visited, value distributions)

## Usage Instructions

### Training the Agent

```bash
# Run training (headless mode for speed)
python testing.py
```

Training output:
```
🔋 Battery-Aware Environment Initialized
   Initial Battery: 100.0%
   Low Battery Threshold: 20.0%

Training Configuration:
  Headless Mode: True
  Verbose Logging: False
  Episodes: 2000
  Q-table Print Interval: 50

📈 Progress Report - Episode 100/2000
  Average Reward (last 100): -23.45
  Average Length (last 100): 187.2
  Success Rate: 34.50%
  Battery Efficiency: 76.3% avg remaining
  Epsilon: 0.819
  Q-table Size: 142 states

📊 Q-Table Summary (Episode 150):
  Total States Visited: 142
  Non-zero Q-values: 568
  Max Q-value: 45.23
  Min Q-value: -87.45
  Average Q-value: -12.34

✓ Q-table saved successfully to q_table_mir100.pkl
  Saved 142 state entries
📁 Files generated:
   - q_table_mir100.pkl (trained Q-table)
   - training_log.json (detailed training log)
   - training_results.png (training plots)
```

### Running Navigation with Trained Agent

```bash
# Navigate using trained policy
python robot_avoidance_ql.py
```

Navigation output:
```
🤖 MiR100 Robot Navigation using Trained Q-Learning Agent
� Battery-Aware Environment Initialized

�🚀 Starting navigation...
   From: [1.0, 1.0, 0.0]
   To: [2.0, 2.0, 0.0]
   Max steps: 1000

📍 Navigation in progress...
   Step 50: Distance to target = 0.8m, Battery: 94.2%
   Step 100: Distance to target = 0.3m, Battery: 88.7%

📊 Navigation completed!
   Final Status: success
   Total Steps: 127
   Total Reward: 67.85
   🔋 Final Battery: 85.3% (high)
   🎉 SUCCESS! Robot reached the destination!
   ⚡ EFFICIENCY BONUS: Completed with high battery!
   📏 Total distance traveled: 1.89m

📈 Summary of 3 navigation attempts:
   Success Rate: 3/3 (100.0%)
   Average Steps: 142.3
   Average Reward: 71.45
```

### File Structure (Updated)

```
robo-gym/
├── training.py                    # Battery-aware Q-Learning training script
├── robot_avoidance_ql.py          # Navigation with trained agent (battery-aware)
├── battery_mir100_env.py          # Battery-aware environment implementation
├── battery_env_registration.py    # Environment registration helper
├── q_table_mir100.pkl             # Saved Q-table (generated)
├── training_log.json              # Detailed training statistics (generated)
├── training_results.png           # Training plots (generated)
└── MotionRL.md                   # This documentation
```

## Battery System Implementation

### Battery Management Features

The enhanced environment includes a comprehensive battery management system:

#### **Battery State Tracking**
- **Real-time monitoring** of battery level (0-100%)
- **Status classification**: full (>80%), high (50-80%), normal (20-50%), low (5-20%), critical (<5%)
- **Consumption breakdown** per action (linear, angular, idle)

#### **Energy Consumption Model**
```python
# Battery drain calculation per step
linear_drain = abs(linear_velocity) * 0.08     # % per velocity unit
angular_drain = abs(angular_velocity) * 0.03   # % per velocity unit  
idle_drain = 0.005                             # % baseline consumption

total_drain = linear_drain + angular_drain + idle_drain
```

#### **Battery-Aware Rewards**
- **Efficiency incentive**: Lower consumption = higher rewards
- **Low battery penalties**: Increasing penalties as battery depletes
- **Completion bonuses**: Extra rewards for finishing with high battery
- **Depletion termination**: Episode ends if battery reaches 0%

#### **Implementation Location**
- **Environment**: `/home/andy/mysource/robo-gym/battery_mir100_env.py`
- **Registration**: `/home/andy/mysource/robo-gym/robo_gym/__init__.py`
- **Base reward function**: Modified from `/home/andy/mysource/robo-gym/robo_gym/envs/mir100/mir100.py`

### Expected Learning Behavior

With battery constraints, the robot learns to:
1. **Plan efficient paths** (shorter routes preserve battery)
2. **Balance speed vs consumption** (slower movement = longer battery life)
3. **Prioritize task completion** within energy constraints
4. **Develop energy-aware navigation strategies**

---

*This documentation covers the battery-aware Q-Learning implementation for robot navigation and obstacle avoidance using the robo-gym framework. For additional details, refer to the source code comments and robo-gym official documentation.*

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