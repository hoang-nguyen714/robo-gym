# Battery-Enhanced MiR100 Environment
# 
# This file shows how to implement battery drain consumption 
# instead of power consumption in the reward structure

import numpy as np
from robo_gym.envs.mir100.mir100 import ObstacleAvoidanceMir100

class BatteryAwareObstacleAvoidanceMir100(ObstacleAvoidanceMir100):
    """
    Enhanced MiR100 environment with battery drain consumption
    instead of simple power consumption in reward calculation
    """
    
    def __init__(self, rs_address=None, **kwargs):
        super().__init__(rs_address, **kwargs)
        
        # Battery management parameters
        self.initial_battery = 100.0      # Initial battery percentage (100%)
        self.current_battery = self.initial_battery
        self.battery_drain_linear = 0.08  # Battery drain per linear velocity unit per step
        self.battery_drain_angular = 0.03 # Battery drain per angular velocity unit per step  
        self.battery_drain_idle = 0.005   # Battery drain when idle (sensors, computation)
        self.low_battery_threshold = 20.0 # Low battery warning threshold
        self.critical_battery_threshold = 5.0  # Critical battery level
        
        print(f"🔋 Battery-Aware Environment Initialized")
        print(f"   Initial Battery: {self.initial_battery}%")
        print(f"   Low Battery Threshold: {self.low_battery_threshold}%")
    
    def reset(self, *, seed=None, options=None):
        """Reset environment and restore battery to full charge"""
        state, info = super().reset(seed=seed, options=options)
        
        # Reset battery to full charge
        self.current_battery = self.initial_battery
        
        # Add battery info to reset info
        info['battery_level'] = self.current_battery
        info['battery_status'] = 'full'
        
        return state, info
    
    def _calculate_battery_drain(self, action):
        """Calculate battery consumption based on robot actions"""
        linear_vel = abs(action[0])
        angular_vel = abs(action[1])
        
        # Calculate battery drain components
        linear_drain = linear_vel * self.battery_drain_linear
        angular_drain = angular_vel * self.battery_drain_angular
        idle_drain = self.battery_drain_idle
        
        total_drain = linear_drain + angular_drain + idle_drain
        
        return {
            'linear_drain': linear_drain,
            'angular_drain': angular_drain, 
            'idle_drain': idle_drain,
            'total_drain': total_drain
        }
    
    def _get_battery_status(self):
        """Get current battery status"""
        if self.current_battery <= self.critical_battery_threshold:
            return 'critical'
        elif self.current_battery <= self.low_battery_threshold:
            return 'low'
        elif self.current_battery >= 80.0:
            return 'high'
        else:
            return 'normal'
    
    def _reward(self, rs_state, action):
        """Modified reward function with battery drain consumption"""
        reward = 0
        done = False
        info = {}
        
        # Calculate distance to the target (same as original)
        target_coords = np.array([rs_state[0], rs_state[1]])
        mir_coords = np.array([rs_state[3], rs_state[4]])
        euclidean_dist_2d = np.linalg.norm(target_coords - mir_coords, axis=-1)
        
        # Distance-based reward (same as original)
        base_reward = -50 * euclidean_dist_2d
        if self.prev_base_reward is not None:
            reward = base_reward - self.prev_base_reward
        self.prev_base_reward = base_reward
        
        # === BATTERY DRAIN IMPLEMENTATION ===
        
        # Calculate battery consumption
        battery_consumption = self._calculate_battery_drain(action)
        
        # Update battery level
        self.current_battery = max(0.0, self.current_battery - battery_consumption['total_drain'])
        
        # Battery-based reward/penalty
        # 1. Penalty for battery consumption (encourages efficiency)
        battery_penalty = -battery_consumption['total_drain'] * 0.5
        
        # 2. Additional penalty for low battery
        if self.current_battery <= self.low_battery_threshold:
            battery_penalty -= (self.low_battery_threshold - self.current_battery) * 0.2
        
        # 3. Critical penalty for very low battery
        if self.current_battery <= self.critical_battery_threshold:
            battery_penalty -= 10.0  # Heavy penalty for critical battery
        
        # Apply battery penalty to reward
        reward += battery_penalty

        # === Idle / stand-still penalty (consistent with base env) ===
        if abs(action[0]) <= getattr(self, 'idle_linear_threshold', 0.05) and abs(action[1]) <= getattr(self, 'idle_angular_threshold', 0.05):
            self.idle_steps = getattr(self, 'idle_steps', 0) + 1
        else:
            self.idle_steps = 0

        if getattr(self, 'idle_steps', 0) >= getattr(self, 'idle_penalty_apply_after', 1):
            reward -= getattr(self, 'idle_penalty_per_step', 0.5)
            info['idle_penalty_applied'] = True
        else:
            info['idle_penalty_applied'] = False

        # Add idle steps to info for debugging
        info['idle_steps'] = int(getattr(self, 'idle_steps', 0))
        # === end idle penalty ===
        
        # Add battery information to info
        battery_status = self._get_battery_status()
        info.update({
            'battery_level': round(self.current_battery, 2),
            'battery_status': battery_status,
            'battery_consumption': battery_consumption,
            'battery_penalty': battery_penalty
        })
        
        # === END BATTERY IMPLEMENTATION ===
        
        # Collision detection (same as original)
        if not self.real_robot:
            if (
                self._sim_robot_collision(rs_state)
                or self._min_laser_reading_below_threshold(rs_state)
                or self._robot_close_to_sim_obstacle(rs_state)
            ):
                reward = -200.0
                done = True
                info["final_status"] = "collision"
        
        # Success condition (same as original)
        if euclidean_dist_2d < self.distance_threshold:
            reward = 100
            done = True
            info["final_status"] = "success"
            
            # Bonus reward for completing with high battery
            if self.current_battery > 50.0:
                reward += (self.current_battery - 50.0) * 0.5  # Efficiency bonus
        
        # Battery depletion condition (NEW)
        if self.current_battery <= 0.0:
            reward = -150.0  # Penalty for depleting battery
            done = True
            info["final_status"] = "battery_depleted"
        
        # Max steps condition (same as original)
        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info["final_status"] = "max_steps_exceeded"
        
        return reward, done, info

# Usage example - how to integrate this into your training
"""
To use this battery-aware environment in your training:

1. Import the class:
   from battery_mir100_env import BatteryAwareObstacleAvoidanceMir100

2. Register it with gymnasium:
   from gymnasium.envs.registration import register
   
   register(
       id="BatteryObstacleAvoidanceMir100Sim-v0",
       entry_point="battery_mir100_env:BatteryAwareObstacleAvoidanceMir100Sim",
   )

3. Create simulation wrapper (similar to original):
   class BatteryAwareObstacleAvoidanceMir100Sim(Simulation, BatteryAwareObstacleAvoidanceMir100):
       cmd = "roslaunch mir100_robot_server sim_robot_server.launch world_name:=lab_6x8.world gazebo_gui:=true"
       
       def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
           Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
           BatteryAwareObstacleAvoidanceMir100.__init__(self, rs_address=self.robot_server_ip, **kwargs)

4. Use in training.py:
   env = gym.make('BatteryObstacleAvoidanceMir100Sim-v0', ip=target_machine_ip, gui=False)
"""