from environment import *
from replaybuffer import *
from actor_stochastic import *
from torch.nn import SmoothL1Loss
import matplotlib.pyplot as plt
import pandas as pd
from critic import *
import time
import torch as T

class Agent:
    def __init__(self,batch_size=32,fillsize=50,tau=0.001,gamma=0.99,lr_pi=1e-4,lr_cp=1e-2,lr_critic=1e-3,ymax=0.5,epochs=50,freq=20,sigma=0.1,units=32,fill_custom=False):
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.epochs = epochs
        self.batch_size = batch_size
        self.fillsize = fillsize
        self.freq = freq
        self.sigma = sigma
        self.sigma_init = sigma
        self.PI = Actor_Stochastic(lr=lr_pi,ymax=ymax,units=units)
        self.critic1 = Critic(lr=lr_critic,units=units)
        self.critic2 = Critic(lr=lr_critic,units=units)
        self.target_critic1 = Critic(lr=lr_critic,units=units)
        self.target_critic2 = Critic(lr=lr_critic,units=units)
        self.target_PI = Actor_Stochastic(lr=lr_pi,ymax=ymax,units=units)
        self.env = Environment()
        self.state = self.env.reset()
        self.terminate = False
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.target_PI.load_state_dict(self.PI.state_dict())
        self.buffer = ReplayBuffer(maxlen=100000)
        self.lossfn = SmoothL1Loss()
        self.losses_policy = []
        self.losses_critic = []
        self.env_steps = 0
        self.episodes = 0
        self.violations = 0
        self.episodic_violations = [0]
        self.policy_violations = 0
        self.episodic_policy_violations = [0]
        self.rewards = 0
        self.episodic_returns = [0]
        self.episodic_returns_test = [0]
        self.episodic_violations_test = [0]
        # initialize the replay buffer
        self.fill_buffer(fill_custom)
    
    def fill_buffer(self,fill_custom=False):
        print('filling buffer....')
        env = Environment()
        state = env.reset()
        terminate = False
        while len(self.buffer) < self.fillsize:
            action = self.PI.sample_action(state,det=False,sigma=0.2)
            nstate,nstate_alt,reward,cost,done,terminate = env.step(action,custom=False)
            if (fill_custom and cost<0):
                done = np.array(True).reshape(1,-1)
            self.buffer.append((state,action,reward,cost,nstate,done))
            if (terminate):
                state = env.reset()
                terminate = False
            else:
                state = nstate
        print(f'buffer size: {len(self.buffer.buffer)}')
    
    def update_target_network(self):
        """
        update the target network parameters using polyak averaging
        """
        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.PI.parameters(), self.target_PI.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)        
    
    def step(self):
        """
        step through environment and save in buffer
        """
        action = self.PI.sample_action(self.state,det=False,sigma=self.sigma)
        nstate,nstate_alt,reward,cost,done,terminate = self.env.step(action,custom=True)
        if (cost<0):
            done = np.array(True).reshape(1,-1)
        self.buffer.append((self.state,action,reward,cost,nstate_alt,done))
        self.env_steps += 1
        self.rewards += reward[0,0]
        if (cost<0):
            self.violations += 1
        if (terminate):
            self.state = self.env.reset()
            self.terminate = False
            self.episodes += 1
            self.episodic_violations.append(self.violations)
            self.violations = 0
            self.episodic_returns.append(self.rewards)
            self.rewards = 0
        else:
            self.state = nstate
    
    
    def test_agent(self):
        """
        test agent using deterministic policy
        """
        env = Environment()
        state = env.reset_test()
        terminate = False
        viols = 0
        rewards = 0
        while not terminate:
            action = self.PI.sample_action(state,det=True)
            nstate,reward,cost,done,terminate = env.step_test(action)
            if (cost<0):
                viols += 1
                #print(viols)
            rewards += reward[0,0]
            state = nstate
        self.episodic_violations_test.append(viols)
        self.episodic_returns_test.append(rewards)
    
    def update_critic_ddpg(self):
        """
        update the critic, used by ddpg
        """
        self.critic1.optimizer.zero_grad()
        s,a,r,c,ns,done = self.buffer.sample(self.batch_size)
        with T.no_grad():
            next_action = self.target_PI.sample_action(T.from_numpy(ns).float(),det=True)
            target_q,_ = self.target_critic1(T.from_numpy(ns).float(),T.from_numpy(next_action).float())
            target_value = T.from_numpy(r).float() + self.gamma*target_q*(1-T.from_numpy(done).float())
        current_q,current_c = self.critic1(T.from_numpy(s).float(),T.from_numpy(a).float())
        c = T.from_numpy(c).float()
        loss = self.lossfn(current_q,target_value)+T.mean(T.clamp(1-current_c*c,min=0))
        self.losses_critic.append(loss.item())
        loss.backward()
        self.critic1.optimizer.step()
    
    
    def update_policy_ddpg(self):
        """
        update the policy
        """
        s,a,r,c,ns,done = self.buffer.sample(self.batch_size)
        self.PI.optimizer.zero_grad()
        action = self.PI.forward(T.from_numpy(s).float())
        Q,_ = self.critic1(T.from_numpy(s).float(),action)
        loss = -Q.mean()
        self.losses_policy.append(loss.item())
        loss.backward()
        self.PI.optimizer.step()
    
    
    def train_ddpg(self):
        """
        train the ddpg agent
        """
        print('training ddpg')
        while (self.episodes<self.epochs):
            self.sigma = max(self.sigma_init/2,self.sigma_init*(1-1.5*self.episodes/self.epochs))
            self.step()
            self.update_critic_ddpg()
            self.update_policy_ddpg()
            self.update_target_network()
            if (self.env_steps%self.freq==0):
                print(f'environment steps: {self.env_steps}, episodes: {self.episodes}, episode violations: {self.episodic_violations[-1]}, episode rewards: {self.episodic_returns[-1]:.2f}, test_reward: {self.episodic_returns_test[-1]:.2f}, test_violations: {self.episodic_violations_test[-1]}')
            if ((self.env_steps+1)%2000==0):
                self.test_agent()
        self.test_agent()
    