import random
from collections import deque
import pickle
import numpy as np

class ReplayBuffer:
    def __init__(self,maxlen) -> None:
        """
        Buffer. 
        """
        self.buffer = deque(maxlen=maxlen)
        
    
    def append(self,exp):
        """
        Appends experience tuple to the replay buffer.
        """
        self.buffer.append(exp)
    
    def sample(self,batch_size):
        """
        Returns a batch of experience. Experience is not normalized.
        """
        sample = random.sample(self.buffer,batch_size)
        sample = zip(*sample)
        sample = [np.concatenate(x) for x in sample]
        return sample

    def __len__(self):
        return len(self.buffer)

"""
#### test the buffer implementation
buff = ReplayBuffer(maxlen=1000,batchsize=64,algo='reteach')
# sample from the buffer with a batch size of 64
state,Qac,Pac,reward,flag = buff.sample(64)
"""