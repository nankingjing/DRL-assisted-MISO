# coding :utf-8

__author__ = 'HYL'
__version__ = '1.0.0'


import random
import numpy as np
import torch
import torch.nn as nn
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
        Pt = 10
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

        self.linear5 = nn.Linear(input_size, hidden_size)
        self.linear6 = nn.Linear(hidden_size, hidden_size)
        self.linear7 = nn.Linear(hidden_size, hidden_size)
        self.linear8 = nn.Linear(hidden_size, output_size)

    def forward(self, s, a):
        sa = torch.cat([s, a], 1)

        q1 = self.linear1(sa)
        q1 = self.linear2(q1)
        q1 = self.linear3(q1)
        q1 = self.linear4(q1)

        q2 = self.linear5(sa)
        q2 = self.linear6(q2)
        q2 = self.linear7(q2)
        q2 = self.linear8(q2)

        return q1, q2
    def Q1(self, s, a):
        sa = torch.cat([s, a], 1)
        q1 = self.linear1(sa)
        q1 = self.linear2(q1)
        q1 = self.linear3(q1)
        q1 = self.linear4(q1)
        return q1

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
        self.policy_noise = 0.2*2*np.pi
        self.noise_clip = 0.5*2*np.pi
        self.policy_freq = 2
        self.total_it = 0
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
        self.total_it += 1
        samples = random.sample(self.buffer, self.batch_size)
        s0, a0, r1, s1 = zip(*samples)
        s0 = torch.tensor(s0, dtype=torch.float)
        a0 = torch.tensor(a0, dtype=torch.float)
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size,-1)
        s1 = torch.tensor(s1, dtype=torch.float)
        
        def critic_learn():
            noise = (torch.randn_like(a0) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            a1 = (self.actor_target(s1) + noise).clamp(0, 2*np.pi)
            target_Q1, target_Q2 = self.critic_target(s1, a1)
            target_Q = r1 + self.gamma * torch.min(target_Q1, target_Q2)
            current_Q1, current_Q2 = self.critic(s0, a0)
            loss_fn = nn.MSELoss()
            critic_loss = loss_fn(current_Q1, target_Q) + loss_fn(current_Q2, target_Q)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
            lmbda = lambda epoch: 0.000001
            self.critic_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.critic_optim, lr_lambda=lmbda)
            self.critic_scheduler.step()
            
        def actor_learn():
            loss = -self.critic.Q1(s0, self.actor(s0)).mean()
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()
            lmbda = lambda epoch: 0.000001
            self.actor_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.actor_optim, lr_lambda=lmbda)
            self.actor_scheduler.step()
                                
        def soft_update(net_target, net, tau):
            for target_param, param  in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
        critic_learn()
        if self.total_it % self.policy_freq == 0:
            actor_learn()
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)