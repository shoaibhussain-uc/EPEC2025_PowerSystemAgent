import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim


class Critic(nn.Module):
    def __init__(self, lr=1e-3,units=64):
        super(Critic, self).__init__()
        # state has 7 dimensions, action has 2 dimensions
        # output of this critic has same dimension 1
        self.statedims = 7
        self.actiondims = 2
        self.lr = lr
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.fc1 = nn.Linear(self.statedims + self.actiondims, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, 1)
        self.fc4 = nn.Linear(units, 1)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
    
    def forward(self, state, action):
        if (type(state) != T.Tensor):
            state = T.tensor(state).float().to(self.device).reshape(-1,self.statedims)
        if (type(action) != T.Tensor):
            action = T.tensor(action).float().to(self.device).reshape(-1,self.actiondims)
        x = T.cat((state, action), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        c = T.tanh(self.fc4(x))*1.5
        return q,c
