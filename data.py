import numpy as np

buses = range(4)
B = {
    (0,1): 0.1,
    (1,2): 0.1,
    (2,3): 0.1,
    (0,3): 0.1
}

#  N is the number of time steps
N = 400
sbase = 100

"""
loads = {
    1: (25+45*np.random.rand(N+1))/sbase,
    2: (25+45*np.random.rand(N+1))/sbase,
    3: (40+50*np.random.rand(N+1))/sbase
}
"""
loads = {
    1: (22.5*np.sin(np.linspace(0,2*2*np.pi,N))+45/2+25)/sbase+np.linspace(0,2,N)/sbase,
    2: (22.5*np.sin(np.linspace(0,2*2*np.pi,N))+45/2+25)/sbase+np.linspace(0,2,N)/sbase,
    3: (25*np.sin(np.linspace(0,2*2*np.pi,N))+25+40)/sbase+np.linspace(0,2,N)/sbase
}


lims = {
    (0,1): 50/sbase,
    (1,2): 50/sbase,
    (2,3): 50/sbase,
    (0,3): 100/sbase
}

gen_lims = {
    1: 50/sbase,
    2: 50/sbase
}

