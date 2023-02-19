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
env = Environment(max_steps=50)

def main():
    actor = Actor(OBSERVATION_SIZE, ACTION_NUM, ACTOR_LR, MAX_ACTIONS)
    actor.load_state_dict(torch.load('models/17-feb-training-no-clipping_actor.pht'))
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

    test(td3)


    

def test(td3: TD3):

    episode = 1

    while True:
        state, _ = env.reset()
        episode_reward = 0

        while True:
            action = td3.forward(state)
            action = mapAction(action)

            next_state, reward, terminated, truncated, _ = env.step(action)

            state = next_state
            episode_reward += reward

            if terminated or truncated:
                break

        episode += 1

        env.get_logger().info(f"Episode: {episode} Reward: {episode_reward}")

def mapAction(tensor):

    tensor[0] = (tensor[0] + 1 ) / 2 * 3 # Map between 0 and 3
    # tensor[1] = tensor[1] - 0.5 # Map between -0.5 and 0.5

    return tensor