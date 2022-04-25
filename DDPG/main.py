# coding :utf-8

__author__ = 'HYL'
__version__ = '1.0.0'

import gym
import numpy

from ddpg import Agent
EPISODES = 5000    # 进行多少个Episode N
STEPS = 20000       # 每个Episode进行多少step T 20000
if __name__ == '__main__':

    env = gym.make('MyEnv-v0')
    env.reset()

    params = {
        'env': env,
        'gamma': 0.99,
        'actor_lr': 0.001,
        'critic_lr': 0.001,
        'tau': 0.001,# 用于target网络软更新的参数
        'capacity': 100000,
        'batch_size': 16,
    }
    
    agent = Agent(**params)
    fo = open("testlog.txt", "w")
    for episode in range(EPISODES):
        s0 = env.reset()
        episode_reward = 0

        for step in range(STEPS):
         
            a0 = agent.act(s0)
            a0 = numpy.clip(a0, 0, 2*numpy.pi)
            s1, r1, done, _ = env.step(a0)
            agent.put(s0[:, 0], a0, r1, s1[:, 0])
            agent.learn()
            episode_reward += r1
            s0 = s1
            st = numpy.str_(step) + ' ' + numpy.str_(episode_reward/(step+1)) + ' ' + numpy.str_(r1)
            fo.write(st+'\n') 
            print(st)
        s = numpy.str_(episode) + ': ' + numpy.str_(episode_reward/STEPS)
        fo.write(s+'\n')
        print(s)
    fo.close()