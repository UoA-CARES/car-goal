from .environment import Environment
from .Actor import Actor
from .Critic import Critic

import rclpy
import time 
from ament_index_python.packages import get_package_share_directory
import random
import torch
import numpy as np
from cares_reinforcement_learning.networks import TD3
from cares_reinforcement_learning.util import MemoryBuffer, Plot

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("Working with GPU")
else:
    DEVICE = torch.device('cpu')
    print("Working with CPU")

BUFFER_CAPACITY = 100_000

GAMMA = 0.995
TAU = 0.005

ACTOR_LR = 1e-4
CRITIC_LR = 1e-3

EPISODE_NUM = 10_000
BATCH_SIZE = 100

MAX_ACTIONS = np.array([3, 1])
MIN_ACTIONS = np.array([0, -1])

OBSERVATION_SIZE = 8
ACTION_NUM = 2

rclpy.init()
env = Environment(max_steps=15)

def main():

    memory = MemoryBuffer(BUFFER_CAPACITY)

    actor = Actor(OBSERVATION_SIZE, ACTION_NUM, ACTOR_LR, MAX_ACTIONS)
    critic_one = Critic(OBSERVATION_SIZE, ACTION_NUM, CRITIC_LR)
    critic_two = Critic(OBSERVATION_SIZE, ACTION_NUM, CRITIC_LR)

    td3 = TD3(
        actor_network=actor,
        critic_one=critic_one,
        critic_two=critic_two,
        max_actions=MAX_ACTIONS,
        min_actions=MIN_ACTIONS,
        gamma=GAMMA,
        tau=TAU,
        device=DEVICE
    )


    env.get_logger().info(f"Filling Buffer...")

    fill_buffer(memory)

    env.get_logger().info(f"Buffer Filled!")

    env.get_logger().info(f"Training Beginning")
    train(td3, memory)


def train(td3, memory: MemoryBuffer):
    plot = Plot(plot_freq=100, checkpoint_freq=20)

    for episode in range(0, EPISODE_NUM):

        state, _ = env.reset()
        episode_reward = 0

        while True:

            # Select an Action
            action = td3.forward(state)
            action = mapAction(action)
            noise = np.random.normal(0, scale=0.1, size=ACTION_NUM)
            noise = np.multiply(noise,  MAX_ACTIONS)
            action = action + noise
            action = np.clip(action, MIN_ACTIONS, MAX_ACTIONS)

            next_state, reward, terminated, truncated, info = env.step(action)
            env.get_logger().info(f"\nNext State: {next_state}, \nAction: {action}, \nReward: {reward}, \nTerminated: {terminated}, \nTruncated: {truncated}\nInfo: {info}") 

            memory.add(state, action, reward, next_state, terminated)

            experiences = memory.sample(BATCH_SIZE)

            for _ in range(0, 1):
                td3.learn(experiences)

            state = next_state
            episode_reward += reward

            if terminated or truncated:
                break
        
        plot.post(episode_reward)
        env.get_logger().info(f"Episode: {episode} Reward: {episode_reward}")
    
    td3.save_models('16-feb-training')
    plot.save_plot('16-feb-training')
    plot.save_csv('16-feb-training')



def fill_buffer(memory):
    while len(memory.buffer) < memory.buffer.maxlen // 10:

        state, _ = env.reset()

        while True:
            env.get_logger().info(f"Buffer Size: {len(memory.buffer)}")
            action = [random.uniform(MIN_ACTIONS[0], MAX_ACTIONS[0]), random.uniform(MIN_ACTIONS[1], MAX_ACTIONS[1])]

            next_state, reward, terminated, truncated, info = env.step(action)
            env.get_logger().info(f"\nNext State: {next_state}, \nAction: {action}, \nReward: {reward}, \nTerminated: {terminated}, \nTruncated: {truncated}\nInfo: {info}") 
            memory.add(state, action, reward, next_state, terminated)

            state = next_state

            if terminated or truncated:
                break

def mapAction(tensor):

    tensor[0] = (tensor[0] + 1 ) / 2 * 3 # Map between 0 and 3
    # tensor[1] = tensor[1] - 0.5 # Map between -0.5 and 0.5

    return tensor