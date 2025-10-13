#!/usr/bin/env python3
"""
Robot Navigation using Trained Q-Learning Agent
==============================================

This script loads a pre-trained Q-Learning model and uses it to navigate
the MiR100 robot from a starting point to a destination while avoiding obstacles.
"""

import gymnasium as gym
import robo_gym
import numpy as np
import pickle
import os
import time
from collections import defaultdict

# Configuration parameters
target_machine_ip = '127.0.0.1'  # or other machine 'xxx.xxx.xxx.xxx'

# Navigation points (should match training configuration)
START_POINT = [0.5, 0.0, 0.0]  # [x, y, yaw] - Starting position (match training)
DESTINATION_POINT = [2.5, 2.5, 0.0]  # [x, y, yaw] - Destination position (match training)

# State discretization parameters (must match training)
POSITION_BINS = 20  # Number of bins for x, y coordinates
LASER_BINS = 5  # Number of bins for laser readings
MAX_DISTANCE = 5.0  # Maximum distance for normalization

# Navigation parameters
MAX_NAVIGATION_STEPS = 1000  # Maximum steps for navigation
VISUALIZATION_DELAY = 0.1  # Delay between steps for visualization (seconds)

class TrainedQLearningAgent:
    """Q-Learning agent that loads and uses a pre-trained Q-table for navigation"""
    
    def __init__(self, q_table_file='q_table_mir100.pkl'):
        """
        Initialize the trained Q-Learning agent
        
        Args:
            q_table_file (str): Path to the saved Q-table file
        """
        self.q_table_file = q_table_file
        self.q_table = defaultdict(lambda: np.zeros(self.get_action_space_size()))
        
        # Define discrete actions (must match training)
        self.discrete_actions = self._create_discrete_actions()
        
        # Load the trained Q-table
        self.load_q_table()
        
    def get_action_space_size(self):
        """Get the size of the discrete action space"""
        return len(self.discrete_actions)
    
    def _create_discrete_actions(self):
        """Create discrete action set from continuous action space (must match training)"""
        actions = []
        linear_speeds = [-1.0, -0.5, 0.0, 0.5, 1.0]
        angular_speeds = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        for lin in linear_speeds:
            for ang in angular_speeds:
                actions.append([lin, ang])
        
        return np.array(actions)
    
    def discretize_state(self, state):
        """Convert continuous state to discrete state for Q-table lookup (must match training)"""
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
    
    def get_best_action(self, state):
        """Get the best action for a given state using the trained Q-table"""
        discrete_state = self.discretize_state(state)
        
        # Choose action with highest Q-value (pure exploitation)
        action_idx = np.argmax(self.q_table[discrete_state])
        
        return action_idx, self.discrete_actions[action_idx]
    
    def load_q_table(self):
        """Load Q-table from file"""
        if os.path.exists(self.q_table_file):
            try:
                with open(self.q_table_file, 'rb') as f:
                    q_table_dict = pickle.load(f)
                    self.q_table = defaultdict(lambda: np.zeros(self.get_action_space_size()), q_table_dict)
                print(f"✓ Successfully loaded Q-table from {self.q_table_file}")
                print(f"  Q-table contains {len(q_table_dict)} state entries")
                return True
            except Exception as e:
                print(f"✗ Error loading Q-table from {self.q_table_file}: {e}")
                print("  Using random policy instead")
                return False
        else:
            print(f"✗ Q-table file {self.q_table_file} not found")
            print("  Please train the model first by running testing.py")
            print("  Using random policy instead")
            return False

def navigate_robot(agent, env, start_point, destination_point, max_steps=MAX_NAVIGATION_STEPS):
    """
    Navigate robot from start to destination using trained Q-Learning agent
    
    Args:
        agent: Trained Q-Learning agent
        env: Gymnasium environment
        start_point: Starting position [x, y, yaw]
        destination_point: Destination position [x, y, yaw]
        max_steps: Maximum navigation steps
        
    Returns:
        dict: Navigation results
    """
    print(f"\n🚀 Starting navigation...")
    print(f"   From: {start_point}")
    print(f"   To: {destination_point}")
    print(f"   Max steps: {max_steps}")
    
    # Reset environment with specified positions
    state, info = env.reset(options={
        'start_pose': start_point,
        'target_pose': destination_point
    })
    
    # Navigation tracking
    total_reward = 0
    step_count = 0
    done = False
    trajectory = []
    
    print(f"\n📍 Navigation in progress...")
    
    while not done and step_count < max_steps:
        # Get robot's current position for trajectory tracking
        robot_pos = [state[0], state[1], state[2]]  # x, y, yaw
        trajectory.append(robot_pos.copy())
        
        # Get best action from trained agent
        action_idx, action = agent.get_best_action(state)
        
        # Execute action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Update tracking
        total_reward += reward
        step_count += 1
        state = next_state
        
        # Print progress every 50 steps
        if step_count % 50 == 0:
            distance_to_target = np.sqrt((state[2] - state[0])**2 + (state[3] - state[1])**2)
            print(f"   Step {step_count}: Distance to target = {distance_to_target:.2f}m")
        
        # Add visualization delay
        time.sleep(VISUALIZATION_DELAY)
    
    # Determine final status
    final_status = info.get('final_status', 'unknown')
    
    # Print results
    print(f"\n📊 Navigation completed!")
    print(f"   Final Status: {final_status}")
    print(f"   Total Steps: {step_count}")
    print(f"   Total Reward: {total_reward:.2f}")
    
    if final_status == 'success':
        print(f"   🎉 SUCCESS! Robot reached the destination!")
    elif final_status == 'collision':
        print(f"   💥 COLLISION! Robot hit an obstacle.")
    elif final_status == 'max_steps_exceeded':
        print(f"   ⏰ TIMEOUT! Maximum steps exceeded.")
    else:
        print(f"   ❓ Navigation ended with status: {final_status}")
    
    # Calculate trajectory statistics
    if len(trajectory) > 1:
        total_distance = 0
        for i in range(1, len(trajectory)):
            dx = trajectory[i][0] - trajectory[i-1][0]
            dy = trajectory[i][1] - trajectory[i-1][1]
            total_distance += np.sqrt(dx**2 + dy**2)
        
        print(f"   📏 Total distance traveled: {total_distance:.2f}m")
    
    return {
        'success': final_status == 'success',
        'final_status': final_status,
        'steps': step_count,
        'reward': total_reward,
        'trajectory': trajectory
    }

def run_multiple_navigations(agent, env, num_runs=5):
    """
    Run multiple navigation attempts to test consistency
    
    Args:
        agent: Trained Q-Learning agent
        env: Gymnasium environment
        num_runs: Number of navigation attempts
    """
    print(f"\n🔄 Running {num_runs} navigation attempts...")
    
    results = []
    success_count = 0
    
    for run in range(num_runs):
        print(f"\n--- Navigation Attempt {run + 1}/{num_runs} ---")
        
        result = navigate_robot(agent, env, START_POINT, DESTINATION_POINT)
        results.append(result)
        
        if result['success']:
            success_count += 1
    
    # Summary statistics
    print(f"\n📈 Summary of {num_runs} navigation attempts:")
    print(f"   Success Rate: {success_count}/{num_runs} ({success_count/num_runs*100:.1f}%)")
    
    if results:
        avg_steps = np.mean([r['steps'] for r in results])
        avg_reward = np.mean([r['reward'] for r in results])
        print(f"   Average Steps: {avg_steps:.1f}")
        print(f"   Average Reward: {avg_reward:.2f}")
    
    return results

def main():
    """Main function to run robot navigation"""
    print("=" * 60)
    print("🤖 MiR100 Robot Navigation using Trained Q-Learning Agent")
    print("=" * 60)
    
    try:
        # Initialize environment
        print(f"\n🌍 Initializing environment...")
        env = gym.make('ObstacleAvoidanceMir100Sim-v0', ip=target_machine_ip, gui=True)
        print(f"✓ Environment initialized successfully")
        
        # Load trained agent
        print(f"\n🧠 Loading trained Q-Learning agent...")
        agent = TrainedQLearningAgent('q_table_mir100.pkl')
        
        # Single navigation run
        print(f"\n" + "="*50)
        print(f"🎯 Single Navigation Run")
        print(f"="*50)
        result = navigate_robot(agent, env, START_POINT, DESTINATION_POINT)
        
        # Multiple navigation runs for testing consistency
        print(f"\n" + "="*50)
        print(f"📊 Multiple Navigation Test")
        print(f"="*50)
        run_multiple_navigations(agent, env, num_runs=3)
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Navigation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during navigation: {e}")
    finally:
        # Clean up
        try:
            env.close()
            print(f"\n🔚 Environment closed successfully")
        except:
            pass
    
    print(f"\n✅ Navigation session completed")

if __name__ == "__main__":
    main()