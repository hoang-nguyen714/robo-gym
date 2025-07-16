import gymnasium as gym
import robo_gym

target_machine_ip = '127.0.0.1' # or other machine 'xxx.xxx.xxx.xxx'

# initialize environment
env = gym.make('ObstacleAvoidanceMir100Sim-v0', ip=target_machine_ip, gui=True)

num_episodes = 20

for episode in range(num_episodes):
    done = False
    env.reset()
    while not done:
        # random step in the environment
        state, reward, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated