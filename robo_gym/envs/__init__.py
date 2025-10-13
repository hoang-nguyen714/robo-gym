# Example
from robo_gym.envs.example.example_env import ExampleEnvSim, ExampleEnvRob

# MiR100
from robo_gym.envs.mir100.mir100 import (
    NoObstacleNavigationMir100Sim,
    NoObstacleNavigationMir100Rob,
)
from robo_gym.envs.mir100.mir100 import (
    ObstacleAvoidanceMir100Sim,
    ObstacleAvoidanceMir100Rob,
)

# Battery-aware MiR100 Environment
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from battery_mir100_env import BatteryAwareObstacleAvoidanceMir100
from robo_gym.envs.simulation_wrapper import Simulation

class BatteryAwareObstacleAvoidanceMir100Sim(Simulation, BatteryAwareObstacleAvoidanceMir100):
    cmd = "roslaunch mir100_robot_server sim_robot_server.launch world_name:=lab_6x8.world gazebo_gui:=true"

    def __init__(
        self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs
    ):
        Simulation.__init__(
            self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs
        )
        BatteryAwareObstacleAvoidanceMir100.__init__(
            self, rs_address=self.robot_server_ip, **kwargs
        )

# UR
from robo_gym.envs.ur.ur_base_env import EmptyEnvironmentURSim, EmptyEnvironmentURRob
from robo_gym.envs.ur.ur_ee_positioning import (
    EndEffectorPositioningURSim,
    EndEffectorPositioningURRob,
)
from robo_gym.envs.ur.ur_avoidance_basic import BasicAvoidanceURSim, BasicAvoidanceURRob
from robo_gym.envs.ur.ur_avoidance_raad import (
    AvoidanceRaad2022URSim,
    AvoidanceRaad2022URRob,
)
from robo_gym.envs.ur.ur_avoidance_raad import (
    AvoidanceRaad2022TestURSim,
    AvoidanceRaad2022TestURRob,
)

from robo_gym.envs.ur.ur_base import EmptyEnvironment2URSim, EmptyEnvironment2URRob
from robo_gym.envs.ur.ur_ee_pos import (
    EndEffectorPositioning2URSim,
    EndEffectorPositioning2URRob,
)
from robo_gym.envs.ur.ur_isaac_reach import IsaacReachURSim, IsaacReachURRob

from robo_gym.envs.panda.panda_base import (
    EmptyEnvironmentPandaSim,
    EmptyEnvironmentPandaRob,
)
from robo_gym.envs.panda.panda_ee_pos import (
    EndEffectorPositioningPandaSim,
    EndEffectorPositioningPandaRob,
)
from robo_gym.envs.panda.panda_isaac_reach import IsaacReachPandaSim, IsaacReachPandaRob
