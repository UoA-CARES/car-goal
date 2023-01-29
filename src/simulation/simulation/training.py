from .environment import Environment

import rclpy
import time 
from ament_index_python.packages import get_package_share_directory
import random
import torch
import torch.nn as nn
import torch.optim as optim
from cares_reinforcement_learning.networks import TD3
from cares_reinforcement_learning.util import MemoryBuffer

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("Working with GPU")
else:
    DEVICE = torch.device('cpu')
    print("Working with CPU")

BUFFER_CAPACITY = 50

GAMMA = 0.995
TAU = 0.005

ACTOR_LR = 1e-4
CRITIC_LR = 1e-3

EPISODE_NUM = 100
BATCH_SIZE = 12

rclpy.init()
env = Environment(max_steps=10)

def main():
    

    

    observation_size = 6
    action_num = 2

    max_actions = [3, 1]
    min_actions = [0, -1]

    memory = MemoryBuffer(BUFFER_CAPACITY)

    actor = Actor(observation_size, action_num, ACTOR_LR, max_actions)
    critic_one = Critic(observation_size, action_num, CRITIC_LR)
    critic_two = Critic(observation_size, action_num, CRITIC_LR)

    td3 = TD3(
        actor_network=actor,
        critic_one=critic_one,
        critic_two=critic_two,
        max_actions=max_actions,
        min_actions=min_actions,
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
    historical_reward = []

    for episode in range(0, EPISODE_NUM):

        state, _ = env.reset()
        episode_reward = 0

        while True:

            # Select an Action
            action = td3.forward(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            env.get_logger().info(f"\nNext State: {next_state}, \nAction: {action}, \nReward: {reward}, \nTerminated: {terminated}, \nTruncated: {truncated}\nInfo: {info}") 

            memory.add(state, action, reward, next_state, terminated)

            experiences = memory.sample(BATCH_SIZE)

            for _ in range(0, 10):
                td3.learn(experiences)

            state = next_state
            episode_reward += reward

            if terminated or truncated:
                break

        historical_reward.append(episode_reward)
        env.get_logger().info(f"Episode: {episode} Reward: {episode_reward}")



def fill_buffer(memory):
    while len(memory.buffer) < memory.buffer.maxlen:

        state, _ = env.reset()

        while True:

            action = [random.uniform(0, 3), random.uniform(-1, 1)]

            next_state, reward, terminated, truncated, _ = env.step(action)

            memory.add(state, action, reward, next_state, terminated)

            state = next_state

            if terminated or truncated:
                break




class Actor(nn.Module):
    def __init__(self, observation_size, num_actions, learning_rate, max_action):
        super(Actor, self).__init__()

        self.max_action = max_action

        self.hidden_size = [128, 64, 32]

        self.h_linear_1 = nn.Linear(in_features=observation_size, out_features=self.hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=self.hidden_size[0], out_features=self.hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=self.hidden_size[1], out_features=self.hidden_size[2])
        self.h_linear_4 = nn.Linear(in_features=self.hidden_size[2], out_features=num_actions)

        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = torch.relu(self.h_linear_1(state))
        x = torch.relu(self.h_linear_2(x))
        x = torch.relu(self.h_linear_3(x))
        x = torch.tanh(self.h_linear_4(x)) 
        return x


class Critic(nn.Module):
    def __init__(self, observation_size, num_actions, learning_rate):
        super(Critic, self).__init__()

        self.hidden_size = [128, 64, 32]

        self.Q1 = nn.Sequential(
            nn.Linear(observation_size + num_actions, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], self.hidden_size[2]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[2], 1)
        )

        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        q1 = self.Q1(x)
        return q1

if __name__ == '__main__':
    main()
