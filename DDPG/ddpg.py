# coding :utf-8

__author__ = 'HYL'
__version__ = '1.0.0'

import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
N=4
M=4
K=4
class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)
        
    def forward(self, s):
        x = self.linear1(s)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        # 归一化层
        # pt=s[2*M*K+N+2*N*M+N*K:2*M*K+N+2*N*M+2*N*K+K].detach().numpy()
        # Pt = sum(pt**2)
        Pt = 20
        x1 = x[:M*K].detach().numpy()
        x1 = (x1-np.mean(x1))/np.sqrt(np.var(x1)) * np.sqrt(Pt**2/(M*K))
        x[:M*K]= torch.from_numpy(x1) 
        x = torch.tanh(x)
        return x

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)

    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)

        return x


class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        s_dim = self.env.observation_space.shape[0]
        a_dim = self.env.action_space.shape[0]

        self.actor = Actor(s_dim, 400, a_dim)
        self.actor_target = Actor(s_dim, 400, a_dim)
        self.critic = Critic(s_dim+a_dim, 400, 1)
        self.critic_target = Critic(s_dim+a_dim, 400, 1)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr = self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = self.critic_lr)
        self.buffer = []
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
    def act(self, s0):
        s0 = torch.tensor(s0[:,0], dtype=torch.float).unsqueeze(0)
        a0 = self.actor(s0).squeeze(0).detach().numpy()
        return a0
    
    def put(self, *transition): 
        if len(self.buffer)== self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)
    
    def learn(self):
        if len(self.buffer) < self.batch_size:
            return 
        
        samples = random.sample(self.buffer, self.batch_size)
        
        s0, a0, r1, s1 = zip(*samples)
        
        s0 = torch.tensor(s0, dtype=torch.float)
        a0 = torch.tensor(a0, dtype=torch.float)
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size,-1)
        s1 = torch.tensor(s1, dtype=torch.float)
        
        def critic_learn():
            a1 = self.actor_target(s1).detach()
            y_true = r1 + self.gamma * self.critic_target(s1, a1).detach()
            y_pred = self.critic(s0, a0)
            loss_fn = nn.MSELoss()
            loss = loss_fn(y_pred, y_true)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()
            lmbda = lambda epoch: 0.00001
            self.critic_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.critic_optim, lr_lambda=lmbda)
            self.critic_scheduler.step()
            
            
            
        def actor_learn():
            loss = -torch.mean( self.critic(s0, self.actor(s0)) )
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()
            lmbda = lambda epoch: 0.00001
            self.actor_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.actor_optim, lr_lambda=lmbda)
            self.actor_scheduler.step()
            
            
                                           
        def soft_update(net_target, net, tau):
            for target_param, param  in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
        critic_learn()
        actor_learn()
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)