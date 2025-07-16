import os

import gymnasium as gym
import robo_gym
import pytest

@pytest.fixture(scope='module')
def env(request):
    ip = os.environ.get("ROBOGYM_SERVERS_HOST", 'robot-servers')
    env = gym.make('ObstacleAvoidanceMir100Sim-v0', ip=ip)
    yield env
    # Access the underlying environment to call kill_sim()
    if hasattr(env, 'kill_sim'):
        env.kill_sim()
    elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'kill_sim'):
        env.unwrapped.kill_sim()

@pytest.mark.commit 
def test_initialization(env):
    env.reset()
    done = False
    for _ in range(10):
        if not done:
            action = env.action_space.sample()
            observation, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    assert env.observation_space.contains(observation)