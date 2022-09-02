import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from Environments.grid_world import GridWorld
from torch.utils.tensorboard import SummaryWriter
import time
#Hyperparameters
learning_rate = 0.0002
gamma         = 0.98

class Policy(nn.Module):
    def __init__(self, in_shape, out_shape):
        super(Policy, self).__init__()
        self.data = []
        
        self.fc1 = nn.Linear(in_shape, 128)
        self.fc2 = nn.Linear(128, out_shape)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x
      
    def put_data(self, item):
        self.data.append(item)
    
    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -torch.log(prob) * R
            loss.backward(retain_graph=False)
        self.data = []
        self.optimizer.step()
        return loss.item()
    
    def train_net3(self):
        R = 0
        for r, prob in self.data[::-1]:
            self.optimizer.zero_grad()
            R = r + gamma * R
            loss = -torch.log(prob) * R
            loss.backward(retain_graph=False)
            self.optimizer.step()
        self.data = []
        return loss.item()
    
    def train_net2(self):
        # G = r0 + yr1 + y^r2 + ..
        # G0 = 0.99^0 * r0
        # G1 = 0.99^1 * r1 + 0.99^0 * r0 
        self.optimizer.zero_grad()
        for T in range(len(self.data)):
            G = 0
            for i, (r, prob) in enumerate(self.data[T::]):
                G += gamma ** i * r
            prob = self.data[T][1]
            loss = -torch.log(prob) * G
            loss.backward()
        self.optimizer.step()
        self.data = []
        return loss.item()
            
                

def main():
    summary = SummaryWriter()
    env = GridWorld(grid_size=5, goal_reward=10, max_step=200)
    obs_shape = 1
    num_actions = env.action_space.n

    pi = Policy(obs_shape, num_actions)
    score = 0.0
    print_interval = 20
    
    
    for n_epi in range(10000):
        s = env.reset()
        done = False
        loss = 0.0
        while not done: 
            prob = pi(torch.from_numpy(np.array([s])).float())
            m = Categorical(prob)
            a = m.sample()
            s_prime, r, done, info = env.step(a.item())
            pi.put_data((r,prob[a]))
            s = s_prime
            score += r
            #env.render()
            
        loss = pi.train_net2()
        if n_epi%print_interval==0 and n_epi!=0:
            avg_score = score/print_interval
            summary.add_scalar('train/loss', loss, n_epi)
            summary.add_scalar('train/avg_score', score, n_epi)
            print("# of episode :{}, avg score : {}".format(n_epi, avg_score))
            score = 0.0
        torch.save(pi.state_dict(), 'policy.pt')
    env.close()

def eval():

    summary = SummaryWriter()
    env = GridWorld(grid_size=5, goal_reward=10, max_step=200)
    obs_shape = 1
    num_actions = env.action_space.n

    pi = Policy(obs_shape, num_actions)
    score = 0.0
    print_interval = 20
    state_dict = torch.load('policy.pt')
    pi.load_state_dict(state_dict)

    for n_epi in range(1000):
        s = env.reset()
        done = False
        while not done: 
            prob = pi(torch.from_numpy(np.array([s])).float())
            m = Categorical(prob)
            a = m.sample()
            s_prime, r, done, info = env.step(a.item())
            s = s_prime
            score += r
            time.sleep(0.2)
            env.render()
        if n_epi%print_interval==0 and n_epi!=0:
            avg_score = score/print_interval
            summary.add_scalar('eval/avg_score', score, n_epi)
            print("# of episode :{}, avg score : {}".format(n_epi, avg_score))
            score = 0.0

    
if __name__ == '__main__':
    main()
    eval()