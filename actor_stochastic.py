import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

class Actor_Stochastic(nn.Module):
    def __init__(self, lr=1e-4,ymax=1.0,units=64):
        super(Actor_Stochastic, self).__init__()
        self.lr = lr
        self.statedims = 7
        self.actiondims = 2
        self.ymax = ymax
        self.fc1 = nn.Linear(self.statedims,units)
        self.fc2 = nn.Linear(units,units)
        self.fcmu = nn.Linear(units, self.actiondims)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        if (type(state) != T.Tensor):
            state = T.tensor(state, dtype=T.float).to(self.device).reshape(-1,self.statedims)
        x = F.relu(self.fc1(state.float()))
        x = F.relu(self.fc2(x))
        mu = T.sigmoid(self.fcmu(x))*self.ymax
        return mu

    def sample_action(self, state,sigma=0.1,det=False):
        mu = self.forward(state)
        dist = Normal(mu,sigma)
        if det:
            action = mu
        else:
            action = dist.sample()
        #print("Raw sampled action:", action)  # Debug print
        #action = T.sigmoid(action) * self.ymax
        #print("Final action:", action)  # Debug print
        action = action.clamp(0, self.ymax)
        return action.detach().cpu().numpy()

    
"""
pi = Actor(ymax=0.5)
sample = pi.sample_action(np.random.rand(7))
print(sample)
print(pi.log_prob(np.random.rand(7),sample))
"""