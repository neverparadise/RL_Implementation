import gym
from gym.spaces import Discrete

import random
import os
import time
import numpy as np


env_config = {"grid_size": 7,
             "goal_reward": 20,
             "out_reward": -10,
             "step_reward":-1,
             "obstacle_reward": -5,
             "max_step": 200 }

class Environment:

    def __init__(self, *args, **kwargs) -> None:
        self.grid_size = env_config['grid_size']
        self.goal_reward = env_config['goal_reward']
        self.max_step = env_config['max_step']
        self.out_reward = env_config["out_reward"]
        self.step_reward = env_config["step_reward"]
        self.obstacle_reward = env_config["obstacle_reward"]
        self.action_space = Discrete(4)
        self.observation_space = Discrete(self.grid_size*self.grid_size)
        self.seeker, self.goal = (0, 0), (self.grid_size-1, self.grid_size-1)
        self.info = {'seeker': self.seeker, 'goal': self.goal}
        self.timestep = 0
        self.obstacle_states = [
            (i, j) for i in range(self.grid_size) for j in range(self.grid_size)
            if i % 2 == 1 and j % 2 == 0
        ]
    def reset(self):
        self.seeker = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)) # row, col
        self.goal = (self.grid_size-1, self.grid_size-1)
        self.timestep = 0
        return self.get_observation()
    
    def get_observation(self):
        return self.grid_size * self.seeker[0] + self.seeker[1]
    
    def get_reward(self):
        reward = self.obstacle_reward if self.seeker in self.obstacle_states else 0
        reward += self.goal_reward if self.seeker == self.goal else 0
        return reward
    
    def is_done(self):
        if self.timestep == self.max_step:
            return True
        return self.seeker == self.goal

    def check_pos(self, seeker):
        is_out = False
        if seeker[0] < 0 or seeker[0] > self.grid_size - 1 or \
            seeker[1] < 0 or seeker[1] > self.grid_size - 1: 
            is_out = True
        return is_out

    def step(self, action):
        self.timestep += 1
        reward = 0
        is_out = False

        if action == 0: # move left
            self.seeker = (self.seeker[0], self.seeker[1] - 1)
            is_out =  self.check_pos(self.seeker)
            if is_out:
                self.seeker = (self.seeker[0], self.seeker[1] + 1)

        elif action == 1: # move right
            self.seeker = (self.seeker[0], self.seeker[1] + 1)
            is_out =  self.check_pos(self.seeker)
            if is_out:
                self.seeker = (self.seeker[0], self.seeker[1] - 1)

        elif action == 2: # move up
            self.seeker = (self.seeker[0] - 1, self.seeker[1])
            is_out =  self.check_pos(self.seeker)
            if is_out:
                self.seeker = (self.seeker[0] + 1, self.seeker[1])
                
        elif action == 3: # move down
            self.seeker = (self.seeker[0] + 1, self.seeker[1])
            is_out =  self.check_pos(self.seeker)
            if is_out:
                self.seeker = (self.seeker[0] - 1, self.seeker[1])
        else:
            raise ValueError("Invalid action")

        if is_out:
            reward = self.out_reward
        else:
            reward = self.get_reward()

        return self.get_observation(), reward, self.is_done(), self.info

    def render(self, *args, **kwaargs):
        os.system('cls' if os.name == 'nt' else 'clear')
        grid_row = ['| ' for _ in range(self.grid_size)] 
        grid = [grid_row + ["|\n"] for _ in range(self.grid_size)]
        grid[self.goal[0]][self.goal[1]] = '|G'
        grid[self.seeker[0]][self.seeker[1]] = '|A'
        for obstacle in self.obstacle_states:
            grid[obstacle[0]][obstacle[1]] = '|O'

        print(''.join([''.join(grid_row) for grid_row in grid]))


class HardGridWorld(Environment, gym.Env):
    def __init__(self, grid_size=5, goal_reward=1, max_step=200, *args, **kwargs) -> None:
        super().__init__(grid_size, goal_reward, max_step, *args, **kwargs)

#gym_env = GymEnvironment()

