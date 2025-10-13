from gymnasium.envs.registration import register

# naming convention: EnvnameRobotSim

# Example Environments
register(
    id="ExampleEnvSim-v0",
    entry_point="robo_gym.envs:ExampleEnvSim",
)

register(
    id="ExampleEnvRob-v0",
    entry_point="robo_gym.envs:ExampleEnvRob",
)

# MiR100 Environments
register(
    id="NoObstacleNavigationMir100Sim-v0",
    entry_point="robo_gym.envs:NoObstacleNavigationMir100Sim",
)

register(
    id="NoObstacleNavigationMir100Rob-v0",
    entry_point="robo_gym.envs:NoObstacleNavigationMir100Rob",
)

register(
    id="ObstacleAvoidanceMir100Sim-v0",
    entry_point="robo_gym.envs:ObstacleAvoidanceMir100Sim",
)

register(
    id="ObstacleAvoidanceMir100Rob-v0",
    entry_point="robo_gym.envs:ObstacleAvoidanceMir100Rob",
)

register(
    id="BatteryObstacleAvoidanceMir100Sim-v0",
    entry_point="robo_gym.envs:BatteryAwareObstacleAvoidanceMir100Sim",
)

# UR Environments
register(
    id="EmptyEnvironmentURSim-v0",
    entry_point="robo_gym.envs:EmptyEnvironmentURSim",
)

register(
    id="EmptyEnvironmentURRob-v0",
    entry_point="robo_gym.envs:EmptyEnvironmentURRob",
)

register(
    id="EndEffectorPositioningURSim-v0",
    entry_point="robo_gym.envs:EndEffectorPositioningURSim",
)

register(
    id="EndEffectorPositioningURRob-v0",
    entry_point="robo_gym.envs:EndEffectorPositioningURRob",
)

register(
    id="BasicAvoidanceURSim-v0",
    entry_point="robo_gym.envs:BasicAvoidanceURSim",
)

register(
    id="BasicAvoidanceURRob-v0",
    entry_point="robo_gym.envs:BasicAvoidanceURRob",
)

register(
    id="AvoidanceRaad2022URSim-v0",
    entry_point="robo_gym.envs:AvoidanceRaad2022URSim",
)

register(
    id="AvoidanceRaad2022URRob-v0",
    entry_point="robo_gym.envs:AvoidanceRaad2022URRob",
)

register(
    id="AvoidanceRaad2022TestURSim-v0",
    entry_point="robo_gym.envs:AvoidanceRaad2022TestURSim",
)

register(
    id="AvoidanceRaad2022TestURRob-v0",
    entry_point="robo_gym.envs:AvoidanceRaad2022TestURRob",
)

# TODO register the following as v1 or v2 of the corresponding v0 ones above instead?
register(
    id="EmptyEnvironment2URSim-v0",
    entry_point="robo_gym.envs:EmptyEnvironment2URSim",
)

register(
    id="EmptyEnvironment2URRob-v0",
    entry_point="robo_gym.envs:EmptyEnvironment2URRob",
)

register(
    id="EndEffectorPositioning2URSim-v0",
    entry_point="robo_gym.envs:EndEffectorPositioning2URSim",
)

register(
    id="EndEffectorPositioning2URRob-v0",
    entry_point="robo_gym.envs:EndEffectorPositioning2URRob",
)

register(
    id="IsaacReachURSim-v0",
    entry_point="robo_gym.envs:IsaacReachURSim",
)

register(
    id="IsaacReachURRob-v0",
    entry_point="robo_gym.envs:IsaacReachURRob",
)

# Panda Environments
register(
    id="EmptyEnvironmentPandaSim-v0",
    entry_point="robo_gym.envs:EmptyEnvironmentPandaSim",
)

register(
    id="EmptyEnvironmentPandaRob-v0",
    entry_point="robo_gym.envs:EmptyEnvironmentPandaRob",
)


register(
    id="EndEffectorPositioningPandaSim-v0",
    entry_point="robo_gym.envs:EndEffectorPositioningPandaSim",
)

register(
    id="EndEffectorPositioningPandaRob-v0",
    entry_point="robo_gym.envs:EndEffectorPositioningPandaRob",
)
register(
    id="IsaacReachPandaSim-v0",
    entry_point="robo_gym.envs:IsaacReachPandaSim",
)

register(
    id="IsaacReachPandaRob-v0",
    entry_point="robo_gym.envs:IsaacReachPandaRob",
)
