import os

import gymnasium as gym
import robo_gym
import math
import numpy as np
import pytest

ur_models = [
    pytest.param("ur3", marks=pytest.mark.nightly),
    pytest.param("ur3e", marks=pytest.mark.nightly),
    pytest.param("ur5", marks=pytest.mark.commit),
    pytest.param("ur5e", marks=pytest.mark.nightly),
    pytest.param("ur10", marks=pytest.mark.nightly),
    pytest.param("ur10e", marks=pytest.mark.nightly),
    pytest.param("ur16e", marks=pytest.mark.nightly),
]


@pytest.fixture(scope="module", params=ur_models)
def env(request):
    ip = os.environ.get("ROBOGYM_SERVERS_HOST", "robot-servers")
    env = gym.make("BasicAvoidanceURSim-v0", ip=ip, ur_model=request.param)
    env.request_param = request.param
    yield env
    # Access the underlying environment to call kill_sim()
    if hasattr(env, 'kill_sim'):
        env.kill_sim()
    elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'kill_sim'):
        env.unwrapped.kill_sim()


@pytest.mark.commit
def test_initialization(env):
    assert env.unwrapped.ur.model_key == env.request_param
    env.reset()
    done = False
    env.step([0, 0, 0, 0, 0])
    for _ in range(10):
        if not done:
            action = env.action_space.sample()
            observation, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    assert env.observation_space.contains(observation)


@pytest.mark.skip(reason="This fails only in CI")
@pytest.mark.flaky(reruns=3)
def test_object_collision(env):
    params = {
        "ur5": {
            "joint_positions": [0.0, -1.57, 1.57, -1.57, 0.0, 0.0],
            "object_coords": [0.2, -0.1, 0.52],
            "action": [-1, 0, 0, 0, 0],
            "n_steps": 30,
        },
    }

    env.reset(
        options={
            "desired_joint_positions": params[env.unwrapped.ur.model_key]["joint_positions"],
            "fixed_object_position": params[env.unwrapped.ur.model_key]["object_coords"],
        }
    )
    done = False
    i = 0
    while (not done) or i <= params[env.unwrapped.ur.model_key]["n_steps"]:
        _, _, terminated, truncated, info = env.step(params[env.unwrapped.ur.model_key]["action"])
        done = terminated or truncated
        i += 1
    assert info["final_status"] == "collision"


@pytest.mark.nightly
def test_reset_joint_positions(env):
    joint_positions = [0.5, -2.7, 1.3, -1.7, -1.9, 1.6]

    state, _ = env.reset(options={"joint_positions": joint_positions})
    assert np.isclose(
        env.unwrapped.ur.normalize_joint_values(joint_positions), state[3:9], atol=0.1
    ).all()


@pytest.mark.commit
def test_object_coordinates(env):

    params = {
        # ? robot up-right, target_coord_in_ee_frame 0.0, -0.3, 0.2, coordinates of target calculated using official dimensions from DH parameters.
        # ? first value is d4+d6
        # ? second value is: d1+a2+a3+d5
        "ur3": {
            "joint_positions": [0.0, -1.57, 0.0, -1.57, 0.0, 0.0],
            "object_coords": [0.0, (0.194 + 0.2), (0.692 + 0.3), 0.0, 0.0, 0.0],
            "polar_coords": {"r": 0.360, "theta": 0.983, "phi": -1.571},
        },
        "ur3e": {
            "joint_positions": [0.0, -1.57, 0.0, -1.57, 0.0, 0.0],
            "object_coords": [0.0, (0.223 + 0.2), (0.694 + 0.3), 0.0, 0.0, 0.0],
            "polar_coords": {"r": 0.360, "theta": 0.983, "phi": -1.571},
        },
        "ur5": {
            "joint_positions": [0.0, -1.57, 0.0, -1.57, 0.0, 0.0],
            "object_coords": [0.0, (0.191 + 0.2), (1.001 + 0.3), 0.0, 0.0, 0.0],
            "polar_coords": {"r": 0.360, "theta": 0.983, "phi": -1.571},
        },
        "ur5e": {
            "joint_positions": [0.0, -1.57, 0.0, -1.57, 0.0, 0.0],
            "object_coords": [0.0, (0.233 + 0.2), (1.079 + 0.3), 0.0, 0.0, 0.0],
            "polar_coords": {"r": 0.360, "theta": 0.983, "phi": -1.571},
        },
        "ur10": {
            "joint_positions": [0.0, -1.57, 0.0, -1.57, 0.0, 0.0],
            "object_coords": [0.0, (0.256 + 0.2), (1.428 + 0.3), 0.0, 0.0, 0.0],
            "polar_coords": {"r": 0.360, "theta": 0.983, "phi": -1.571},
        },
        "ur10e": {
            "joint_positions": [0.0, -1.57, 0.0, -1.57, 0.0, 0.0],
            "object_coords": [0.0, (0.291 + 0.2), (1.485 + 0.3), 0.0, 0.0, 0.0],
            "polar_coords": {"r": 0.360, "theta": 0.983, "phi": -1.571},
        },
        "ur16e": {
            "joint_positions": [0.0, -1.57, 0.0, -1.57, 0.0, 0.0],
            "object_coords": [0.0, (0.291 + 0.2), (1.139 + 0.3), 0.0, 0.0, 0.0],
            "polar_coords": {"r": 0.360, "theta": 0.983, "phi": -1.571},
        },
    }

    state, _ = env.reset(
        options={
            "joint_positions": params[env.unwrapped.ur.model_key]["joint_positions"],
            "fixed_object_position": params[env.unwrapped.ur.model_key]["object_coords"],
        }
    )
    assert np.isclose(
        [params[env.unwrapped.ur.model_key]["polar_coords"]["r"]], state[0], atol=0.05
    ).all()
    assert np.isclose(
        [
            params[env.unwrapped.ur.model_key]["polar_coords"]["theta"],
            params[env.unwrapped.ur.model_key]["polar_coords"]["phi"],
        ],
        state[1:3],
        atol=0.2,
    ).all()


test_ur_fixed_joints = [
    (
        "BasicAvoidanceURSim-v0",
        True,
        False,
        False,
        False,
        False,
        False,
        "ur3",
    ),  # fixed shoulder_pan
    (
        "BasicAvoidanceURSim-v0",
        False,
        True,
        False,
        False,
        False,
        False,
        "ur3e",
    ),  # fixed shoulder_lift
    (
        "BasicAvoidanceURSim-v0",
        False,
        False,
        False,
        False,
        False,
        True,
        "ur5",
    ),  # fixed wrist_3
    (
        "BasicAvoidanceURSim-v0",
        True,
        False,
        True,
        False,
        False,
        False,
        "ur5e",
    ),  # fixed Base and Elbow
    (
        "BasicAvoidanceURSim-v0",
        False,
        False,
        True,
        False,
        False,
        False,
        "ur10",
    ),  # fixed elbow
    (
        "BasicAvoidanceURSim-v0",
        False,
        False,
        False,
        True,
        False,
        False,
        "ur10e",
    ),  # fixed wrist_1
    (
        "BasicAvoidanceURSim-v0",
        False,
        False,
        False,
        False,
        True,
        False,
        "ur16e",
    ),  # fixed wrist_2
]


@pytest.mark.nightly
@pytest.mark.parametrize(
    "env_name, fix_base, fix_shoulder, fix_elbow, fix_wrist_1, fix_wrist_2, fix_wrist_3, ur_model",
    test_ur_fixed_joints,
)
@pytest.mark.flaky(reruns=3)
def test_fixed_joints(
    env_name,
    fix_base,
    fix_shoulder,
    fix_elbow,
    fix_wrist_1,
    fix_wrist_2,
    fix_wrist_3,
    ur_model,
):
    env = gym.make(
        env_name,
        ip="robot-servers",
        fix_base=fix_base,
        fix_shoulder=fix_shoulder,
        fix_elbow=fix_elbow,
        fix_wrist_1=fix_wrist_1,
        fix_wrist_2=fix_wrist_2,
        fix_wrist_3=fix_wrist_3,
        ur_model=ur_model,
    )
    state, _ = env.reset()
    initial_joint_positions = state[3:9]
    # Take 20 actions
    action = env.action_space.sample()
    for _ in range(20):
        state, _, _, _, _ = env.step(action)
    joint_positions = state[3:9]

    if fix_base:
        assert math.isclose(
            initial_joint_positions[0], joint_positions[0], abs_tol=0.05
        )
    if fix_shoulder:
        assert math.isclose(
            initial_joint_positions[1], joint_positions[1], abs_tol=0.05
        )
    if fix_elbow:
        assert math.isclose(
            initial_joint_positions[2], joint_positions[2], abs_tol=0.05
        )
    if fix_wrist_1:
        assert math.isclose(
            initial_joint_positions[3], joint_positions[3], abs_tol=0.05
        )
    if fix_wrist_2:
        assert math.isclose(
            initial_joint_positions[4], joint_positions[4], abs_tol=0.05
        )
    if fix_wrist_3:
        assert math.isclose(
            initial_joint_positions[5], joint_positions[5], abs_tol=0.05
        )

    # Access the underlying environment to call kill_sim()
    if hasattr(env, 'kill_sim'):
        env.kill_sim()
    elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'kill_sim'):
        env.unwrapped.kill_sim()
