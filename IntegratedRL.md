# IntegratedRL: Two-Stage Hierarchical Reinforcement Learning for Autonomous Robot Navigation

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Stage 1: MotionRL - Navigation Control](#stage-1-motionrl---navigation-control)
4. [Stage 2: PredicateRL - Decision Making](#stage-2-predicaterl---decision-making)
5. [Behavior Trees Integration](#behavior-trees-integration)
6. [Implementation Framework](#implementation-framework)
7. [Training Pipeline](#training-pipeline)
8. [Deployment Architecture](#deployment-architecture)
9. [Performance Metrics](#performance-metrics)
10. [Usage Instructions](#usage-instructions)
11. [Troubleshooting](#troubleshooting)

## Overview

This project implements a **hierarchical two-stage reinforcement learning system** for autonomous robot navigation with intelligent battery management. The system combines **behavior trees** for high-level decision making with **Q-Learning** for low-level control, ensuring safe and efficient robot operations.

### Key Features
- **Two-Stage Learning Architecture**: Separate training for decision making and motion control
- **Hierarchical Control**: High-level predicates control low-level actions
- **Battery-Aware Navigation**: Intelligent energy management with adaptive return-home policies  
- **Behavior Trees Integration**: Structured decision making with condition nodes
- **Safety-First Design**: Collision avoidance and battery depletion prevention
- **Real-time Adaptation**: Dynamic threshold adjustment based on mission requirements

### System Components
- **PredicateRL**: High-level decision maker (WHEN to return home for charging)
- **MotionRL**: Low-level navigation controller (HOW to navigate safely)
- **Behavior Tree**: Executive coordination between components
- **Battery Management**: Real-time energy monitoring and consumption modeling

## System Architecture

### Hierarchical Control Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    BEHAVIOR TREE EXECUTIVE                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────┐ │
│  │  Mission Queue  │    │  Safety Monitor │    │ Battery │ │
│  │   Management    │    │   & Override    │    │ Monitor │ │
│  └─────────────────┘    └─────────────────┘    └─────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
                ▼               ▼               ▼
     ┌─────────────────┐ ┌─────────────┐ ┌──────────────┐
     │  PREDICATERL    │ │  MOTIONRL   │ │   SAFETY     │
     │  (Condition     │ │ (Navigation │ │  CONTROLLER  │
     │   Nodes)        │ │  Control)   │ │              │
     └─────────────────┘ └─────────────┘ └──────────────┘
     │                   │               │
     │ Battery Threshold │ Robot Actions │ Emergency Stop
     │ Decisions         │ [lin_vel,     │ Collision
     │ [15%-40%]         │  ang_vel]     │ Avoidance
     │                   │               │
     └───────────────────┼───────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   MiR100 ROBOT      │
              │  (Gazebo Simulation │
              │   + ROS Noetic)     │
              └─────────────────────┘
```

### Information Flow

1. **Mission Planning**: Behavior tree receives mission targets
2. **Condition Evaluation**: PredicateRL evaluates battery/distance state  
3. **Decision Making**: Condition nodes determine current mode (mission/return home)
4. **Action Generation**: MotionRL generates navigation commands
5. **Safety Validation**: Safety controller validates and potentially overrides actions
6. **Execution**: Commands sent to robot hardware/simulation

## Stage 1: MotionRL - Navigation Control

### Purpose
MotionRL handles **low-level navigation control**, learning how to safely navigate from any starting point to any destination while avoiding obstacles and managing energy consumption.

### Environment Configuration
- **Environment**: `BatteryObstacleAvoidanceMir100Sim-v0`
- **Robot**: MiR100 mobile robot with 16-beam laser scanner
- **World**: Lab environment (6x8m) with static obstacles  
- **Physics**: Gazebo simulation with realistic dynamics

### State Space (MotionRL)
```python
observation_space = [
    robot_x,           # Robot X position [-∞, +∞]
    robot_y,           # Robot Y position [-∞, +∞] 
    robot_yaw,         # Robot orientation [-π, +π]
    target_x,          # Target X position [-∞, +∞]
    target_y,          # Target Y position [-∞, +∞]
    target_yaw,        # Target orientation [-π, +π]
    laser_scan[0:15],  # 16-beam laser readings [0.0, 30.0]
    battery_level      # Current battery percentage [0.0, 100.0]
]
```

**Discretized for Q-Learning**:
- **Position bins**: 15 bins for relative position coordinates
- **Distance bins**: 10 bins for target distance
- **Laser bins**: 3 bins for obstacle proximity (safe/caution/danger)
- **Battery bins**: 20 bins for battery level (5% increments)

### Action Space (MotionRL)
**Discrete Actions**: 20 combinations of [linear_velocity, angular_velocity]
```python
# Optimized for forward navigation
linear_velocities = [0.0, 0.2, 0.5, 1.0]      # Forward-only motion
angular_velocities = [-1.0, -0.5, 0.0, 0.5, 1.0]  # Full turning range
```

### Reward Function (MotionRL)
```python
def calculate_reward(state, action, next_state, info):
    reward = 0.0
    
    # Distance-based reward (encourages progress toward target)
    distance_reward = -50.0 * euclidean_distance(robot_pos, target_pos)
    
    # Battery consumption penalty (encourages efficiency)
    battery_penalty = -0.5 * total_battery_drain
    
    # Low battery warning penalties
    if battery_level < 20.0:
        reward -= 2.0  # Increased penalty for low battery
    if battery_level < 5.0:
        reward -= 10.0  # Critical battery penalty
    
    # Task completion rewards
    if task_completed:
        reward += 100.0  # Success bonus
        if battery_level > 50.0:
            reward += (battery_level - 50) * 0.5  # Efficiency bonus
    
    # Safety penalties
    if collision_detected:
        reward -= 200.0  # Collision penalty
    if min_laser_reading < safety_threshold:
        reward -= 10.0   # Proximity penalty
    
    # Battery depletion termination
    if battery_level <= 0.0:
        reward -= 100.0  # Depletion penalty
    
    return reward
```

### Learning Parameters (MotionRL)
```python
LEARNING_RATE = 0.2        # Q-value update step size
DISCOUNT_FACTOR = 0.95     # Future reward importance
EPSILON = 1.0              # Initial exploration rate
EPSILON_DECAY = 0.999      # Exploration decay rate
EPSILON_MIN = 0.01         # Minimum exploration rate
NUM_EPISODES = 2000        # Training episodes
```

## Stage 2: PredicateRL - Decision Making

### Purpose  
PredicateRL serves as the **high-level decision maker**, learning optimal battery thresholds for returning home to charge. It acts as intelligent condition nodes in the behavior tree structure.

### State Space (PredicateRL)
**Continuous State**: `[normalized_battery_level, normalized_distance_to_dock]`
```python
state_space = [
    battery_level / 100.0,           # Normalized battery [0.0, 1.0]
    min(distance_to_dock / 10.0, 1.0)  # Normalized distance [0.0, 1.0]
]
```

**Discretized for Q-Learning**:
- **Battery bins**: 20 bins (5% increments)
- **Distance bins**: 10 bins (1m increments up to 10m)

### Action Space (PredicateRL)
**Discrete Battery Thresholds**: 6 threshold options
```python
battery_thresholds = [15, 20, 25, 30, 35, 40]  # Percentage levels
```

**Action Interpretation**:
- **Action 0**: Return home when battery ≤ 15% (risky, maximum mission time)
- **Action 1**: Return home when battery ≤ 20% (balanced risk/mission)
- **Action 2**: Return home when battery ≤ 25% (conservative)
- **Action 3**: Return home when battery ≤ 30% (safe)
- **Action 4**: Return home when battery ≤ 35% (very safe)  
- **Action 5**: Return home when battery ≤ 40% (ultra-conservative)

### Reward Function (PredicateRL)
```python
def calculate_predicate_reward(mission_status, battery_status):
    if mission_completed and returned_home_safely:
        return +20.0  # Successful mission completion
    elif returned_home_prematurely:
        return -10.0  # Premature return (mission incomplete)
    elif battery_depleted_before_docking:
        return -100.0 # Failed to reach dock (critical failure)
    else:
        return -1.0   # Step cost (encourages efficiency)
```

### Learning Parameters (PredicateRL)
```python
LEARNING_RATE = 0.1        # Slower learning for strategic decisions
DISCOUNT_FACTOR = 0.95     # Long-term reward focus
EPSILON = 1.0              # Initial exploration rate
EPSILON_DECAY = 0.995      # Slower decay for thorough exploration
EPSILON_MIN = 0.01         # Minimum exploration rate
NUM_EPISODES = 1000        # Training episodes
```

## Behavior Trees Integration

### Behavior Tree Structure

```
                    ROOT (Sequence)
                         │
        ┌────────────────┼────────────────┐
        │                │                │
    SAFETY_CHECK    MISSION_MANAGER   EXECUTION
   (Condition)       (Selector)       (Action)
        │                │                │
        │    ┌───────────┼──────────┐     │
        │    │           │          │     │
        ▼    ▼           ▼          ▼     ▼
   [Battery  [RETURN   [CONTINUE   [LOW   [NAVIGATE
    > 0%]    HOME]     MISSION]   BATTERY] TO_TARGET]
              │          │          │         │
              │          │          │         │ 
              ▼          ▼          ▼         ▼
        [PredicateRL] [Mission   [PredicateRL] [MotionRL]
        [Decision]    Execution] [Decision]   [Control]
              │          │          │         │
              ▼          ▼          ▼         ▼
         [MotionRL]  [MotionRL]  [Warning]  [Robot
         [Navigate   [Navigate   [System]   Commands]
          to Dock]   to Target]     
```

### Behavior Tree Node Types

#### **1. Condition Nodes (PredicateRL)**
```python
class BatteryConditionNode:
    def __init__(self, predicate_agent):
        self.predicate_agent = predicate_agent
    
    def evaluate(self, battery_level, distance_to_dock):
        state = [battery_level/100.0, min(distance_to_dock/10.0, 1.0)]
        threshold_idx = self.predicate_agent.choose_action(state)
        recommended_threshold = [15,20,25,30,35,40][threshold_idx]
        
        return battery_level <= recommended_threshold
```

#### **2. Action Nodes (MotionRL)**
```python
class NavigationActionNode:
    def __init__(self, motion_agent, target_type="mission"):
        self.motion_agent = motion_agent
        self.target_type = target_type
    
    def execute(self, current_state, target_position):
        action_idx, action = self.motion_agent.choose_action(
            current_state, exploit=True
        )
        return action  # [linear_vel, angular_vel]
```

#### **3. Composite Nodes (Executive Control)**
- **Sequence**: Execute children in order until one fails
- **Selector**: Execute children until one succeeds  
- **Parallel**: Execute multiple children simultaneously

### Behavior Tree Execution Flow

1. **Safety Check**: Verify robot operational status
2. **Battery Evaluation**: PredicateRL assesses current battery/distance state
3. **Mode Selection**:
   - If battery ≤ threshold: Switch to "Return Home" mode
   - If battery > threshold: Continue "Mission" mode
4. **Action Generation**: MotionRL generates navigation commands
5. **Safety Validation**: Verify commands are safe to execute
6. **Command Execution**: Send validated commands to robot

## Implementation Framework

### File Structure
```
robo-gym/
├── MotionRL.md                        # Stage 1 documentation  
├── IntegratedRL.md                    # This comprehensive documentation
├── 
├── # Stage 1: MotionRL Implementation
├── training.py                        # MotionRL Q-Learning training
├── robot_avoidance_ql.py              # MotionRL navigation deployment  
├── battery_mir100_env.py              # Battery-aware environment
├── battery_env_registration.py        # Environment registration
├── 
├── # Stage 2: PredicateRL Implementation (Future)
├── predicate_rl/
│   ├── predicate_env.py              # PredicateRL environment
│   ├── predicate_agent.py            # PredicateRL Q-Learning agent
│   ├── predicate_training.py         # PredicateRL training script
│   └── behavior_tree.py              # Behavior tree implementation
├── 
├── # Integration Components (Future)  
├── integrated_training.py             # Two-stage training pipeline
├── integrated_deployment.py           # Complete system deployment
├── behavior_tree_controller.py        # BT executive controller
├── 
├── # Generated Files
├── q_table_mir100.pkl                 # Trained MotionRL Q-table
├── predicate_q_table.pkl             # Trained PredicateRL Q-table  
├── training_results.png               # Training performance plots
└── integrated_results.png             # System performance analysis
```

### Technology Stack
- **Simulation**: Gazebo + ROS Noetic
- **RL Framework**: Custom Q-Learning with Gymnasium integration
- **Robot Platform**: MiR100 mobile robot
- **Programming**: Python 3.8+
- **Behavior Trees**: Custom implementation or py_trees library
- **Visualization**: Matplotlib, ROS RViz

## Training Pipeline

### Two-Stage Training Process

#### **Phase 1: MotionRL Training (Current)**
```bash
# Stage 1: Train navigation and obstacle avoidance
cd /home/andy/mysource/robo-gym
python training.py

# Expected output:
# - q_table_mir100.pkl (trained navigation agent)
# - training_results.png (performance plots)
# - training_log.json (detailed metrics)
```

**Training Configuration**:
- **Episodes**: 2000
- **Environment**: BatteryObstacleAvoidanceMir100Sim-v0
- **Success Criteria**: >80% success rate, <5% battery depletion
- **Duration**: ~2-4 hours (headless training)

#### **Phase 2: PredicateRL Training (Future Implementation)**
```bash  
# Stage 2: Train battery management decisions
python predicate_rl/predicate_training.py

# Expected output:
# - predicate_q_table.pkl (trained decision agent) 
# - predicate_results.png (decision performance plots)
# - threshold_analysis.json (threshold usage statistics)
```

**Training Configuration**:
- **Episodes**: 1000  
- **Simulation**: Uses trained MotionRL agent for mission execution
- **Success Criteria**: >90% successful mission completion
- **Duration**: ~1-2 hours

#### **Phase 3: Integrated Training (Future Implementation)**
```bash
# Stage 3: Fine-tune integrated system
python integrated_training.py

# Expected output:
# - integrated_performance.json (system metrics)
# - behavior_tree_analysis.png (decision tree visualization)
# - mission_success_report.pdf (comprehensive analysis)
```

### Training Curriculum

#### **MotionRL Curriculum**
1. **Basic Navigation**: Simple point-to-point movement
2. **Obstacle Avoidance**: Navigation with static obstacles
3. **Battery Management**: Energy-efficient path planning  
4. **Multi-target Missions**: Sequential waypoint navigation
5. **Return-to-Dock**: Charging station navigation

#### **PredicateRL Curriculum**  
1. **Single Mission**: Learn thresholds for one-target missions
2. **Multi-Mission**: Optimize for sequential targets
3. **Variable Distance**: Adapt to different dock distances
4. **Mission Priorities**: Balance urgency vs battery safety
5. **Dynamic Environments**: Adapt to changing conditions

## Deployment Architecture

### Integrated System Controller

```python
class IntegratedRLController:
    """
    Main controller integrating PredicateRL and MotionRL
    with Behavior Tree executive coordination
    """
    
    def __init__(self):
        # Load trained agents
        self.motion_agent = MotionRLAgent()
        self.motion_agent.load_q_table('q_table_mir100.pkl')
        
        self.predicate_agent = PredicateRLAgent() 
        self.predicate_agent.load_q_table('predicate_q_table.pkl')
        
        # Initialize behavior tree
        self.behavior_tree = BehaviorTreeController()
        
        # System state
        self.current_mode = "standby"  # standby/mission/returning_home
        self.mission_queue = []
        self.dock_position = [0.0, 0.0, 0.0]
        
    def execute_mission_cycle(self):
        """Main execution loop with behavior tree coordination"""
        while self.mission_queue:
            # Behavior tree tick
            tree_status = self.behavior_tree.tick()
            
            if tree_status == "SUCCESS":
                # Mission completed successfully
                self.handle_mission_completion()
            elif tree_status == "RUNNING":
                # Continue current behavior
                continue  
            elif tree_status == "FAILURE":
                # Handle mission failure
                self.handle_mission_failure()
```

### Real-time Decision Making

```python
def make_navigation_decision(self, robot_state):
    """
    Coordinate PredicateRL and MotionRL for navigation decisions
    """
    # Extract current state information
    battery_level = robot_state['battery_level']
    robot_position = robot_state['position']
    distance_to_dock = calculate_distance(robot_position, self.dock_position)
    
    # PredicateRL: Evaluate if robot should return home
    predicate_state = [
        battery_level / 100.0,
        min(distance_to_dock / 10.0, 1.0)
    ]
    
    threshold_idx = self.predicate_agent.choose_action(predicate_state)
    recommended_threshold = self.predicate_agent.battery_thresholds[threshold_idx]
    
    # Decision: Return home or continue mission?
    if battery_level <= recommended_threshold:
        if self.current_mode != "returning_home":
            self.initiate_return_home()
            return "RETURNING_HOME"
    
    # MotionRL: Generate navigation action
    motion_state = self.prepare_motion_state(robot_state)
    action_idx, action = self.motion_agent.choose_action(motion_state, exploit=True)
    
    return action  # [linear_velocity, angular_velocity]
```

### Safety Integration

```python
class SafetyController:
    """Safety validation and override system"""
    
    def validate_action(self, action, robot_state):
        """Validate and potentially override navigation commands"""
        
        # Emergency battery check
        if robot_state['battery_level'] < 5.0:
            return self.emergency_dock_action(robot_state)
        
        # Collision avoidance check  
        if min(robot_state['laser_readings']) < 0.3:
            return self.collision_avoidance_action(robot_state)
        
        # Velocity limits check
        action = self.enforce_velocity_limits(action)
        
        return action  # Validated or modified action
```

## Performance Metrics

### MotionRL Metrics
- **Success Rate**: Percentage of episodes reaching target safely
- **Average Steps**: Mean steps required for task completion  
- **Battery Efficiency**: Average battery remaining at completion
- **Collision Rate**: Percentage of episodes ending in collision
- **Path Efficiency**: Ratio of optimal path length to actual path length

### PredicateRL Metrics  
- **Decision Accuracy**: Percentage of correct threshold decisions
- **Mission Completion Rate**: Successfully completed missions with safe return
- **Premature Return Rate**: Missions aborted due to conservative thresholds
- **Battery Depletion Rate**: Missions failed due to battery exhaustion  
- **Threshold Distribution**: Usage frequency of different battery thresholds

### Integrated System Metrics
- **Overall Mission Success**: End-to-end mission completion rate
- **System Availability**: Percentage of time robot is operational
- **Energy Efficiency**: Missions completed per full battery cycle
- **Adaptation Performance**: Success rate in novel scenarios
- **Safety Record**: Incidents per operational hour

### Expected Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| MotionRL Success Rate | >95% | High reliability for individual navigation tasks |
| PredicateRL Decision Accuracy | >90% | Correct battery threshold decisions |
| Integrated Mission Success | >85% | End-to-end system performance |
| Battery Depletion Rate | <2% | Safety-first approach |
| Collision Rate | <1% | Critical safety requirement |
| Energy Efficiency | >3 missions/charge | Operational productivity |

## Usage Instructions

### Prerequisites
```bash
# System requirements
Ubuntu 20.04 + ROS Noetic
Python 3.8+
Gazebo 11
robo-gym framework

# Install dependencies
pip install gymnasium numpy matplotlib
pip install py-trees  # For behavior tree implementation
```

### Stage 1: MotionRL Training & Deployment

```bash
# Train MotionRL agent
cd /home/andy/mysource/robo-gym  
python training.py

# Test trained navigation
python robot_avoidance_ql.py
```

### Stage 2: PredicateRL Training (Future)

```bash
# Train PredicateRL agent (requires trained MotionRL)
python predicate_rl/predicate_training.py

# Analyze threshold performance
python predicate_rl/threshold_analysis.py
```

### Stage 3: Integrated Deployment (Future)

```bash
# Deploy integrated system
python integrated_deployment.py

# Monitor system performance  
python system_monitor.py
```

### Configuration Examples

#### **Conservative Mission Profile** 
```python
# High safety, lower productivity
predicate_config = {
    'preferred_thresholds': [30, 35, 40],  # Conservative battery levels
    'mission_priority': 'safety_first',
    'max_missions_per_cycle': 2
}
```

#### **Aggressive Mission Profile**
```python  
# Higher productivity, calculated risks
predicate_config = {
    'preferred_thresholds': [15, 20, 25],  # Aggressive battery usage
    'mission_priority': 'productivity_focused', 
    'max_missions_per_cycle': 4
}
```

#### **Adaptive Mission Profile**
```python
# Dynamic adaptation based on conditions
predicate_config = {
    'adaptive_thresholds': True,
    'mission_priority': 'balanced',
    'context_awareness': True
}
```

## Troubleshooting

### MotionRL Issues

#### **Training Convergence Problems**
```bash
# Symptom: Low success rate after 1000+ episodes
# Solution: Check hyperparameters and reward function

# Debug training progress
grep "Success Rate" training_log.json
python plot_training_progress.py

# Adjust hyperparameters in training.py:
LEARNING_RATE = 0.1      # Reduce if oscillating
EPSILON_DECAY = 0.995    # Slower decay for more exploration
```

#### **Navigation Failures**
```bash
# Symptom: Robot gets stuck or moves inefficiently  
# Solution: Check state discretization and action space

# Monitor robot state
rostopic echo /odom
rostopic echo /scan

# Adjust discretization parameters:
POSITION_BINS = 20       # Increase for finer control
LASER_BINS = 5          # Increase for better obstacle detection
```

### PredicateRL Issues (Future)

#### **Threshold Selection Problems**
- **Conservative bias**: Increase penalty for premature returns
- **Aggressive bias**: Increase penalty for battery depletion
- **Poor adaptation**: Check state representation and exploration

#### **Integration Conflicts**
- **Mode switching errors**: Debug behavior tree logic
- **Action coordination**: Verify MotionRL/PredicateRL handoff
- **Safety override conflicts**: Check safety controller priorities

### System Integration Issues (Future)

#### **Behavior Tree Execution**
```python
# Debug behavior tree state
print(f"Current BT Status: {behavior_tree.get_status()}")
print(f"Active Nodes: {behavior_tree.get_active_nodes()}")
print(f"Failed Nodes: {behavior_tree.get_failed_nodes()}")
```

#### **Performance Monitoring**
```bash
# System health check
python system_health_check.py

# Performance analysis  
python performance_analyzer.py --metric all --duration 24h
```

### Common Error Messages

1. **"PredicateRL agent not found"**
   - Solution: Train PredicateRL first with `python predicate_training.py`

2. **"Behavior tree initialization failed"**  
   - Solution: Check py_trees installation and BT definition files

3. **"MotionRL/PredicateRL version mismatch"**
   - Solution: Retrain both agents with compatible configurations

4. **"Safety controller override loop"**
   - Solution: Check safety thresholds and emergency action definitions

---

## Future Enhancements

### Planned Features
- **Multi-robot coordination**: Fleet-level PredicateRL decisions
- **Dynamic mission prioritization**: Adaptive task scheduling
- **Learning transfer**: Knowledge sharing between robots
- **Predictive maintenance**: Battery degradation modeling
- **Human-robot interaction**: Natural language mission commands

### Research Directions
- **Hierarchical reinforcement learning**: More sophisticated architectures
- **Meta-learning**: Rapid adaptation to new environments  
- **Explainable AI**: Interpretable decision making for safety-critical applications
- **Sim-to-real transfer**: Bridging simulation and real-world deployment

---

*This documentation covers the integrated two-stage reinforcement learning system combining MotionRL for navigation control and PredicateRL for intelligent decision making. The system uses behavior trees to coordinate between components while ensuring safe and efficient autonomous robot operations.*

**System Status**: 
- ✅ **Stage 1 (MotionRL)**: Implemented and documented
- 🔄 **Stage 2 (PredicateRL)**: Architecture designed, implementation pending  
- ⏳ **Integration**: Framework defined, development planned

For technical details on individual components, refer to:
- `MotionRL.md` - Stage 1 implementation details
- Source code comments and robo-gym documentation
- Behavior tree and safety controller specifications (to be developed)