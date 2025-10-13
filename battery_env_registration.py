# Complete Battery-Aware Environment Registration
# Add this to your robo_gym/__init__.py or create a separate registration file

from gymnasium.envs.registration import register
from robo_gym.envs.simulation_wrapper import Simulation
from battery_mir100_env import BatteryAwareObstacleAvoidanceMir100

class BatteryAwareObstacleAvoidanceMir100Sim(Simulation, BatteryAwareObstacleAvoidanceMir100):
    """Battery-aware simulation environment"""
    cmd = "roslaunch mir100_robot_server sim_robot_server.launch world_name:=lab_6x8.world gazebo_gui:=true"
    
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        BatteryAwareObstacleAvoidanceMir100.__init__(self, rs_address=self.robot_server_ip, **kwargs)

# Register the new environment
register(
    id="BatteryObstacleAvoidanceMir100Sim-v0",
    entry_point="__main__:BatteryAwareObstacleAvoidanceMir100Sim",
)