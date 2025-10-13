import gymnasium as gym
import robo_gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import os
import json
from datetime import datetime

# Configuration parameters
target_machine_ip = '127.0.0.1'  # or other machine 'xxx.xxx.xxx.xxx'

# Configurable starting point and destination point
START_POINT = [1.0, 1.0, 0.0]  # [x, y, yaw] - Starting position (closer to destination)
DESTINATION_POINT = [2.0, 2.0, 0.0]  # [x, y, yaw] - Destination position (closer to start)

# Training mode configuration
HEADLESS_TRAINING = True  # Set to False to show Gazebo GUI during training
VERBOSE_LOGGING = False    # Enable detailed logging during training (reduced for performance)
SHOW_GUI_FOR_TESTING = True  # Show GUI when testing trained agent
Q_TABLE_PRINT_INTERVAL = 50  # Print Q-table every N episodes (reduced frequency for performance)

# Q-Learning parameters
LEARNING_RATE = 0.2  # Increased for faster learning
DISCOUNT_FACTOR = 0.95
EPSILON = 1.0  # Initial exploration rate
EPSILON_DECAY = 0.999  # Slower decay for more exploration
EPSILON_MIN = 0.01
NUM_EPISODES = 2000  # More episodes for better convergence

# State discretization parameters
POSITION_BINS = 15  # Reduced bins for coarser discretization (faster learning)
LASER_BINS = 3  # Reduced bins for laser readings (simpler state space)
MAX_DISTANCE = 5.0  # Maximum distance for normalization

class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: dictionary with state as key and action values as values
        self.q_table = defaultdict(lambda: np.zeros(self.get_action_space_size()))
        
        # Define discrete actions for the continuous action space
        self.discrete_actions = self._create_discrete_actions()
        
    def get_action_space_size(self):
        return len(self.discrete_actions)
    
    def _create_discrete_actions(self):
        """Create discrete action set from continuous action space"""
        # Define a set of discrete actions [linear_velocity, angular_velocity]
        actions = []
        # Remove negative linear velocities - only forward motion for better navigation
        linear_speeds = [0.0, 0.2, 0.5, 1.0]  # Only forward motion
        angular_speeds = [-1.0, -0.5, 0.0, 0.5, 1.0]  # Keep all angular velocities for turning
        
        for lin in linear_speeds:
            for ang in angular_speeds:
                actions.append([lin, ang])
        
        return np.array(actions)
    
    def discretize_state(self, state):
        """Convert continuous state to discrete state for Q-table indexing"""
        # Extract relevant features from state
        robot_x = state[0]  # Robot x position
        robot_y = state[1]  # Robot y position
        target_x = state[2]  # Target x position
        target_y = state[3]  # Target y position
        
        # Calculate relative position to target
        rel_x = target_x - robot_x
        rel_y = target_y - robot_y
        distance_to_target = np.sqrt(rel_x**2 + rel_y**2)
        
        # Discretize relative position
        rel_x_bin = min(int((rel_x + MAX_DISTANCE) / (2 * MAX_DISTANCE) * POSITION_BINS), POSITION_BINS - 1)
        rel_y_bin = min(int((rel_y + MAX_DISTANCE) / (2 * MAX_DISTANCE) * POSITION_BINS), POSITION_BINS - 1)
        
        # Discretize distance to target
        distance_bin = min(int(distance_to_target / MAX_DISTANCE * POSITION_BINS), POSITION_BINS - 1)
        
        # Get laser data (last 16 values are laser readings for ObstacleAvoidanceMir100)
        laser_data = state[-16:]  # Get last 16 laser readings
        
        # Find minimum laser reading (closest obstacle)
        min_laser = np.min(laser_data)
        min_laser_bin = min(int(min_laser / 10.0 * LASER_BINS), LASER_BINS - 1)
        
        # Create discrete state tuple
        discrete_state = (rel_x_bin, rel_y_bin, distance_bin, min_laser_bin)
        return discrete_state
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        discrete_state = self.discretize_state(state)
        
        if np.random.random() < self.epsilon:
            # Exploration: choose random action
            action_idx = np.random.choice(len(self.discrete_actions))
        else:
            # Exploitation: choose best action
            action_idx = np.argmax(self.q_table[discrete_state])
        
        return action_idx, self.discrete_actions[action_idx]
    
    def shape_reward(self, state, next_state, original_reward):
        """Add reward shaping to guide robot towards target"""
        # Calculate distances to target
        current_dist = np.sqrt((state[2] - state[0])**2 + (state[3] - state[1])**2)
        next_dist = np.sqrt((next_state[2] - next_state[0])**2 + (next_state[3] - next_state[1])**2)
        
        # Reward for getting closer to target
        distance_reward = (current_dist - next_dist) * 10  # Scale factor for distance improvement
        
        # Penalty for being too far from target
        distance_penalty = -next_dist * 0.1
        
        # Combine original reward with shaped rewards
        shaped_reward = original_reward + distance_reward + distance_penalty
        
        return shaped_reward
    
    def update_q_table(self, state, action_idx, reward, next_state, done):
        """Update Q-table using Q-learning formula"""
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        # Apply reward shaping to help guide learning
        if not done:
            shaped_reward = self.shape_reward(state, next_state, reward)
        else:
            shaped_reward = reward  # Keep terminal rewards unchanged
        
        # Current Q-value
        current_q = self.q_table[discrete_state][action_idx]
        
        if done:
            # Terminal state
            target_q = shaped_reward
        else:
            # Q-learning update rule
            next_max_q = np.max(self.q_table[discrete_next_state])
            target_q = shaped_reward + self.discount_factor * next_max_q
        
        # Update Q-value
        self.q_table[discrete_state][action_idx] += self.learning_rate * (target_q - current_q)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_q_table(self, filename):
        """Save Q-table to file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
            
            with open(filename, 'wb') as f:
                pickle.dump(dict(self.q_table), f)
            print(f"✓ Q-table saved successfully to {filename}")
            print(f"  Saved {len(self.q_table)} state entries")
        except Exception as e:
            print(f"✗ Error saving Q-table to {filename}: {e}")
    
    def load_q_table(self, filename):
        """Load Q-table from file"""
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    q_table_dict = pickle.load(f)
                    self.q_table = defaultdict(lambda: np.zeros(self.get_action_space_size()), q_table_dict)
                print(f"✓ Q-table loaded successfully from {filename}")
                print(f"  Loaded {len(q_table_dict)} state entries")
            except Exception as e:
                print(f"✗ Error loading Q-table from {filename}: {e}")
        else:
            print(f"ℹ No saved Q-table found at {filename} - starting with empty Q-table")
    
    def get_q_table_stats(self):
        """Get statistics about the current Q-table"""
        if not self.q_table:
            return {
                'total_states': 0,
                'non_zero_entries': 0,
                'max_q_value': 0,
                'min_q_value': 0,
                'avg_q_value': 0
            }
        
        total_states = len(self.q_table)
        all_q_values = []
        non_zero_count = 0
        
        for state, q_values in self.q_table.items():
            for q_val in q_values:
                all_q_values.append(q_val)
                if abs(q_val) > 1e-10:  # Consider very small values as zero
                    non_zero_count += 1
        
        all_q_values = np.array(all_q_values)
        
        return {
            'total_states': total_states,
            'non_zero_entries': non_zero_count,
            'max_q_value': float(np.max(all_q_values)) if len(all_q_values) > 0 else 0,
            'min_q_value': float(np.min(all_q_values)) if len(all_q_values) > 0 else 0,
            'avg_q_value': float(np.mean(all_q_values)) if len(all_q_values) > 0 else 0
        }
    
    def print_q_table_summary(self, episode):
        """Print a summary of the Q-table"""
        stats = self.get_q_table_stats()
        print(f"\n📊 Q-Table Summary (Episode {episode}):")
        print(f"  Total States Visited: {stats['total_states']}")
        print(f"  Non-zero Q-values: {stats['non_zero_entries']}")
        print(f"  Max Q-value: {stats['max_q_value']:.4f}")
        print(f"  Min Q-value: {stats['min_q_value']:.4f}")
        print(f"  Average Q-value: {stats['avg_q_value']:.4f}")
        
        # Show a few example states with highest Q-values
        if self.q_table:
            print(f"\n🔍 Top 3 States with Highest Q-values:")
            max_q_states = []
            for state, q_values in self.q_table.items():
                max_q = np.max(q_values)
                max_action = np.argmax(q_values)
                max_q_states.append((state, max_q, max_action))
            
            # Sort by Q-value and show top 3
            max_q_states.sort(key=lambda x: x[1], reverse=True)
            for i, (state, q_val, action) in enumerate(max_q_states[:3]):
                linear_vel, angular_vel = self.discrete_actions[action]
                print(f"  {i+1}. State {state} → Q={q_val:.4f}, Action=[{linear_vel:.1f}, {angular_vel:.1f}]")

def log_episode_details(episode, state, action, reward, next_state, done, info):
    """Log detailed information for each episode step"""
    if VERBOSE_LOGGING:
        robot_pos = [state[0], state[1]]
        target_pos = [state[2], state[3]]
        distance = np.sqrt((target_pos[0] - robot_pos[0])**2 + (target_pos[1] - robot_pos[1])**2)
        
        print(f"Episode {episode+1}: Step taken")
        print(f"  Robot Position: [{robot_pos[0]:.2f}, {robot_pos[1]:.2f}]")
        print(f"  Target Position: [{target_pos[0]:.2f}, {target_pos[1]:.2f}]")
        print(f"  Distance to Target: {distance:.2f}m")
        print(f"  Action: [Linear: {action[0]:.1f}, Angular: {action[1]:.1f}]")
        print(f"  Reward: {reward:.2f}")
        
        # Battery information logging (if available)
        if 'battery_level' in info:
            battery_status = info.get('battery_status', 'unknown')
            battery_consumption = info.get('battery_consumption', {})
            print(f"  🔋 Battery Level: {info['battery_level']:.1f}% ({battery_status})")
            if battery_consumption:
                print(f"     Battery Drain: {battery_consumption.get('total_drain', 0):.4f}%")
        
        if done:
            final_status = info.get('final_status', 'unknown')
            print(f"  ✓ Episode Completed - Status: {final_status}")
            if 'battery_level' in info:
                print(f"  🔋 Final Battery Level: {info['battery_level']:.1f}%")
        print()

def train_q_learning():
    """Main Q-Learning training function"""
    # Initialize environment for training (headless mode)
    print(f"=" * 60)
    print(f"🚀 Q-Learning Training Session")
    print(f"=" * 60)
    print(f"Training Configuration:")
    print(f"  Headless Mode: {HEADLESS_TRAINING}")
    print(f"  Verbose Logging: {VERBOSE_LOGGING}")
    print(f"  Episodes: {NUM_EPISODES}")
    print(f"  Q-table Print Interval: {Q_TABLE_PRINT_INTERVAL}")
    print(f"  Start Point: {START_POINT}")
    print(f"  Destination: {DESTINATION_POINT}")
    print(f"=" * 60)
    
    env = gym.make('BatteryObstacleAvoidanceMir100Sim-v0', 
                   ip=target_machine_ip, 
                   gui=not HEADLESS_TRAINING)  # GUI disabled for headless training
    
    # Initialize Q-Learning agent
    agent = QLearningAgent(
        action_space=env.action_space,
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        epsilon=EPSILON,
        epsilon_decay=EPSILON_DECAY,
        epsilon_min=EPSILON_MIN
    )
    
    # Try to load existing Q-table
    agent.load_q_table('q_table_mir100.pkl')
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print(f"\n🎯 Starting Q-Learning training...")
    training_start_time = datetime.now()
    
    for episode in range(NUM_EPISODES):
        # Print current episode if verbose logging is enabled
        if VERBOSE_LOGGING:
            print(f"\n🔄 Episode {episode + 1}/{NUM_EPISODES} (ε={agent.epsilon:.3f})")
        
        # Reset environment with configured start and target positions
        state, info = env.reset(options={
            'start_pose': START_POINT,
            'target_pose': DESTINATION_POINT
        })
        
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Choose action
            action_idx, action = agent.choose_action(state)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Log episode details if verbose
            if VERBOSE_LOGGING and steps < 5:  # Only log first 5 steps to avoid spam
                log_episode_details(episode, state, action, reward, next_state, done, info)
            
            # Update Q-table
            agent.update_q_table(state, action_idx, reward, next_state, done)
            
            # Update metrics
            total_reward += reward
            steps += 1
            state = next_state
            
            # Check for success
            if terminated and info.get('final_status') == 'success':
                success_count += 1
        
        # Decay exploration rate
        agent.decay_epsilon()
        
        # Store episode metrics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Print Q-table summary every N episodes
        if (episode + 1) % Q_TABLE_PRINT_INTERVAL == 0:
            agent.print_q_table_summary(episode + 1)
        
        # Print progress every 100 episodes or if verbose logging is disabled
        if (episode + 1) % 100 == 0 or (VERBOSE_LOGGING and (episode + 1) % 10 == 0):
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            success_rate = success_count / (episode + 1) * 100
            
            print(f"\n📈 Progress Report - Episode {episode + 1}/{NUM_EPISODES}")
            print(f"  Average Reward (last 100): {avg_reward:.2f}")
            print(f"  Average Length (last 100): {avg_length:.2f}")
            print(f"  Success Rate: {success_rate:.2f}%")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Q-table Size: {len(agent.q_table)} states")
    
    training_end_time = datetime.now()
    training_duration = training_end_time - training_start_time
    
    # Save trained Q-table
    agent.save_q_table('q_table_mir100.pkl')
    
    # Save training log
    training_log = {
        'training_config': {
            'num_episodes': NUM_EPISODES,
            'learning_rate': LEARNING_RATE,
            'discount_factor': DISCOUNT_FACTOR,
            'epsilon_decay': EPSILON_DECAY,
            'headless': HEADLESS_TRAINING,
            'start_point': START_POINT,
            'destination_point': DESTINATION_POINT
        },
        'final_results': {
            'success_count': success_count,
            'final_success_rate': success_count / NUM_EPISODES * 100,
            'final_epsilon': agent.epsilon,
            'q_table_size': len(agent.q_table),
            'training_duration_seconds': training_duration.total_seconds()
        },
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'q_table_stats': agent.get_q_table_stats()
    }
    
    with open('training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"\n🎉 Training Completed!")
    print(f"  Total Duration: {training_duration}")
    print(f"  Final Success Rate: {success_count / NUM_EPISODES * 100:.2f}%")
    print(f"  Final Q-table Size: {len(agent.q_table)} states")
    print(f"  Training log saved to: training_log.json")
    
    # Plot training results
    plot_training_results(episode_rewards, episode_lengths, success_count, NUM_EPISODES)
    
    return agent, env

def plot_training_results(rewards, lengths, success_count, total_episodes):
    """Plot training metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot episode rewards
    ax1.plot(rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    
    # Plot moving average of rewards
    window_size = 100
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(moving_avg)
        ax2.set_title(f'Moving Average Rewards (window={window_size})')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Reward')
        ax2.grid(True)
    
    # Plot episode lengths
    ax3.plot(lengths)
    ax3.set_title('Episode Lengths')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Steps')
    ax3.grid(True)
    
    # Plot success rate
    success_rate = success_count / total_episodes * 100
    ax4.bar(['Success Rate'], [success_rate])
    ax4.set_title(f'Overall Success Rate: {success_rate:.1f}%')
    ax4.set_ylabel('Percentage')
    ax4.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    print(f"📊 Training plots saved to: training_results.png")
    if not HEADLESS_TRAINING:
        plt.show()

def test_trained_agent(agent, env, num_test_episodes=10):
    """Test the trained agent"""
    print(f"\n🧪 Testing trained agent for {num_test_episodes} episodes...")
    
    test_rewards = []
    test_successes = 0
    
    # Set epsilon to 0 for pure exploitation
    agent.epsilon = 0.0
    
    for episode in range(num_test_episodes):
        state, info = env.reset(options={
            'start_pose': START_POINT,
            'target_pose': DESTINATION_POINT
        })
        
        total_reward = 0
        done = False
        steps = 0
        
        while not done:
            action_idx, action = agent.choose_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        test_rewards.append(total_reward)
        
        if terminated and info.get('final_status') == 'success':
            test_successes += 1
            print(f"  Test Episode {episode + 1}: ✅ SUCCESS (Reward: {total_reward:.2f}, Steps: {steps})")
        else:
            status = info.get('final_status', 'unknown')
            print(f"  Test Episode {episode + 1}: ❌ FAILED ({status}) (Reward: {total_reward:.2f}, Steps: {steps})")
    
    avg_test_reward = np.mean(test_rewards)
    test_success_rate = test_successes / num_test_episodes * 100
    
    print(f"\n📊 Test Results:")
    print(f"  Average Reward: {avg_test_reward:.2f}")
    print(f"  Success Rate: {test_success_rate:.1f}%")

if __name__ == "__main__":
    # Train the Q-Learning agent (headless mode)
    agent, training_env = train_q_learning()
    
    # Close training environment
    training_env.close()
    
    # Create a new environment for testing (with GUI if enabled)
    if SHOW_GUI_FOR_TESTING:
        print(f"\n🖥️ Creating test environment with GUI enabled...")
        test_env = gym.make('BatteryObstacleAvoidanceMir100Sim-v0', 
                           ip=target_machine_ip, 
                           gui=True)
        
        # Test the trained agent
        test_trained_agent(agent, test_env)
        
        # Close test environment
        test_env.close()
    else:
        print(f"\n⚠️ Skipping GUI testing (SHOW_GUI_FOR_TESTING = False)")
    
    print(f"\n✅ Training session completed!")
    print(f"📁 Files generated:")
    print(f"   - q_table_mir100.pkl (trained Q-table)")
    print(f"   - training_log.json (detailed training log)")
    print(f"   - training_results.png (training plots)")
    print(f"\n🚀 Use robot_avoidance_ql.py to run navigation with the trained agent!")