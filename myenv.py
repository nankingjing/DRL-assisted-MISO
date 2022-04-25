from gym import spaces, core
import numpy as np
import numpy.matlib
from numpy import random, mat
import gym
# core.Env 是 gym 的环境基类,自定义的环境就是根据自己的需要重写其中的方法；
# 必须要重写的方法有:
# __init__()：构造函数
# reset()：初始化环境
# step()：环境动作,即环境对agent的反馈
# render()：如果要进行可视化则实现


class MyEnv(core.Env):
    def __init__(self):
        # {
        #     'G': numpy.matlib.eye(8, 8),
        #     'theta': np.matlib.identity(8)
        # }
        self.N = 4
        # array number of RIS
        self.M = 4
        #  array number of BS
        self.K = 4
        K = self.K
        M = self.M
        N = self.N
        # self.H1 = random.rayleigh(size=(self.N, self.M))
        # self.hk2 = random.rayleigh(size=(self.N, 1))
        # self.act_dic = {
        #     'G': self.G,
        #     'theta': self.theta
        # }
        # self.obs_dic = {
        #     'G': self.G,
        #     'theta': self.theta,
        #     'H1': self.H1,
        #     'hk2': self.hk2,
        # }
        self.action_space = spaces.Box(
            low=0, high=2*np.pi, shape=(2*M*K+N, 1))  # 动作空间
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(2*M*K+N+2*N*M+2*N*K+K+2*K*K, 1)) # 状态空间

    def reset(self):
        ...
        K = self.K
        M = self.M
        N = self.N
        self.state = np.zeros([2*M*K+N+2*N*M+2*N*K+K+2*K*K, 1])
        s = self.state
        G_r = np.diag(np.diag(np.ones((M, K))))
        G_phi = np.zeros((M, K))
        self.G = G_r * np.exp(1j * G_phi)
        theta_r = np.ones(self.N)
        theta_phi = np.zeros(self.N)
        self.theta = np.diag(theta_r * np.exp(1j * theta_phi))  # Φ
        theta = self.theta
        H1 = self.Rayleighchannel(N,M)
        H1r = H1.real.reshape(N*M,1)
        H1i = H1.imag.reshape(N*M,1)
        h = self.Rayleighchannel(N,K)
        hr = h.real.reshape(N*K,1)
        hi = h.imag.reshape(N*K,1)
        G = self.G
        pt = np.zeros((K,1))
        for k in range(K):
            gk = G[:,k]
            gkt = G[:,k].T
            pt[k] = abs((gkt*gk)[0])
        h_ = h.T.conjugate()*mat(theta)*H1*mat(G)
        h_r = h_.real.reshape(K*K,1)
        h_i = h_.imag.reshape(K*K,1)
        # s[:2*M*K] = mi_to_a(numpy.matlib.eye(self.M, self.K, dtype=complex))
        # s[2*M*K:2*M*K+N] = mi_to_a(numpy.matlib.identity(self.N, dtype=complex))
        # s[2*M*K+2*N*N:2*M*K+2*N*N+N *
        #     M] = m_to_a(random.rayleigh(size=(self.N, self.M)))
        # s[2*M*K+2*N*N+N*M:] = m_to_a(random.rayleigh(size=(self.N, 1)))
        s[:M*K] = G_r.reshape(M*K, 1)
        s[M*K:2*M*K] = G_phi.reshape(M*K, 1)
        s[2*M*K:2*M*K+N] = theta_phi.reshape(N,1)
        s[2*M*K+N:2*M*K+N+N*M] = H1r
        s[2*M*K+N+N*M:2*M*K+N+2*N*M] = H1i
        s[2*M*K+N+2*N*M:2*M*K+N+2*N*M+N*K] = hr
        s[2*M*K+N+2*N*M+N*K:2*M*K+N+2*N*M+2*N*K] = hi
        s[2*M*K+N+2*N*M+2*N*K:2*M*K+N+2*N*M+2*N*K+K] = pt
        s[2*M*K+N+2*N*M+2*N*K+K:2*M*K+N+2*N*M+2*N*K+K+K*K] = h_r
        s[2*M*K+N+2*N*M+2*N*K+K+K*K:2*M*K+N+2*N*M+2*N*K+K+2*K*K] = h_i
        return s

    def step(self, action):
        ...
        reward = self._get_reward()
        done = self._get_done()
        obs = self._get_observation(action)
        info = {}  # 用于记录训练过程中的环境信息,便于观察训练状态
        return obs, reward, done, info

    # 根据需要设计相关辅助函数
    # def action_to_actiongenv(self,action):
    #     ...
    #     K = self.K
    #     M = self.M
    #     N = self.N
    #     actionenv ={
    #         'G': spaces.Box(low=-1,high=1,shape=(self.M, self.K),dtype=complex),
    #         'theta': spaces.Box(low=-1,high=1,shape=(self.N, 1),dtype=complex)
    #     }
    #     actionenv['G'] = ai_to_m(action[:2*M*K],M,K)
    #     actionenv['theta'] = ai_to_m(action[2*M*K:],N,1)
    #     return actionenv
    # def obs_to_obsagent(self,obs):
    #     ...
    #     K = self.K
    #     M = self.M
    #     N = self.N
    #     obsgent = np.zeros([2*M*K+2*N+N*M+N*1,1])
    #     obsgent[:2*M*K] = mi_to_a(obs['G'])
    #     obsgent[2*M*K:2*M*K+2*N] = mi_to_a(obs['theta'])
    #     obsgent[2*M*K+2*N:2*M*K+2*N+N*M] = m_to_a(obs['H1'])
    #     obsgent[2*M*K+2*N+N*M:] = m_to_a(obs['hk2'])
    #     return obsgent

    def _get_observation(self, action):
        ...
        K = self.K
        M = self.M
        N = self.N
        s = self.state
        s[:2*M*K+N][:, 0] = action
        G_r = s[:M*K].reshape(M, K)
        G_phi = s[M*K:2*M*K].reshape(M, K)
        G = G_r * np.exp(1j * G_phi)
        pt = np.zeros((K,1))
        for k in range(K):
            gk = G[:,k]
            gkt = G[:,k].T
            pt[k] = abs((gkt*gk)[0])
        s[2*M*K+N+2*N*M+2*N*K:2*M*K+N+2*N*M+2*N*K+K] = pt
        H1r= s[2*M*K+N:2*M*K+N+N*M]
        H1i= s[2*M*K+N+N*M:2*M*K+N+2*N*M]
        H1 = (H1r + 1j*H1i).reshape(N,M)
        hr= s[2*M*K+N+2*N*M:2*M*K+N+2*N*M+N*K]
        hi= s[2*M*K+N+2*N*M+N*K:2*M*K+N+2*N*M+2*N*K] 
        h = (hr + 1j*hi).reshape(N,K)
        theta_r = np.ones(N)
        theta_phi= s[2*M*K:2*M*K+N].reshape(N,)
        theta = np.diag(theta_r * np.exp(1j * theta_phi))
        h_ = mat(h).T.conjugate()*mat(theta)*mat(H1)*mat(G)
        h_r = h_.real.reshape(K*K,1)
        h_i = h_.imag.reshape(K*K,1)
        s[2*M*K+N+2*N*M+2*N*K+K:2*M*K+N+2*N*M+2*N*K+K+K*K] = h_r
        s[2*M*K+N+2*N*M+2*N*K+K+K*K:2*M*K+N+2*N*M+2*N*K+K+2*K*K] = h_i
        return s

    def Rayleighchannel(self,a,b):
        Kdb = 0
        K = 10**(Kdb/10)
        H_los = np.ones([a, b])
        H_rayleigh = (random.normal(0, 1, size=(a, b))+1j *
                    random.normal(0, 1, size=(a, b)))/np.sqrt(2)
        H = np.sqrt(K/(1+K))*H_los + np.sqrt(1/(1+K))*H_rayleigh
        return H
    def _get_reward(self):
        ...
        K = self.K
        M = self.M
        N = self.N
        p = np.zeros(self.K)
        R = np.zeros(self.K)
        # reward = 0
        s=self.state
        G_r = s[:M*K].reshape(M, K)
        G_phi = s[M*K:2*M*K].reshape(M, K)
        G = G_r * np.exp(1j * G_phi)
        theta_r = np.ones(N)
        theta_phi= s[2*M*K:2*M*K+N].reshape(N,)
        theta = np.diag(theta_r * np.exp(1j * theta_phi))
        H1r= s[2*M*K+N:2*M*K+N+N*M]
        H1i= s[2*M*K+N+N*M:2*M*K+N+2*N*M]
        H1 = (H1r + 1j*H1i).reshape(N,M)
        hr= s[2*M*K+N+2*N*M:2*M*K+N+2*N*M+N*K]
        hi= s[2*M*K+N+2*N*M+N*K:2*M*K+N+2*N*M+2*N*K] 
        h = (hr + 1j*hi).reshape(N,K)
        for k in range(self.K):
            hk2 = h[:, k]
            hk2T = hk2.T.conjugate()
            gk = mat(G)[:, k]
            total = 0
            for n in range(self.K):
                if n != k:
                    gn = mat(G)[:, n]
                    temp = abs((mat(hk2T)*mat(theta)*mat(H1)*gn)[0,0])**2
                    total += temp
                else:
                    total += 0
            p[k] = (abs((mat(hk2T)*mat(theta)*mat(H1)*gk)[0, 0])**2)/(total+1)
            R[k] = np.log2(1+p[k])
            # reward += R[k]
        reward = sum(R)
        return reward

    def _get_done(self):
        ...
        done = False
        return done
