#from mimetypes import init
import numpy as np
from pyomo.environ import *
from data import *


class Environment:
    def __init__(self):
        self.N = N
        self.sbase = sbase
        self.B = B
        self.lims = lims
        self.buses = list(buses)
        self.loads = loads
        self.gen_lims = gen_lims
        print('# define the model')
        self.define_model()
    
    def reset(self,custom=True):
        self.n = 0
        self.terminate = False
        prev_flows = np.array([0.0,0.0,0.0,0.0]).reshape(1,-1)
        current_load = np.array((self.loads[1][self.n],self.loads[2][self.n],self.loads[3][self.n])).reshape(1,-1)
        if custom:
            current_load = np.random.normal(current_load,0.2*current_load).clip(0,current_load)
        init_state = np.concatenate((prev_flows,current_load),axis=1)
        self.state = init_state
        #self.prev_cost = np.array(0).reshape(1,-1)
        return self.state

    def reset_test(self):
        self.n = 0
        self.terminate = False
        prev_flows = np.array([0.0,0.0,0.0,0.0]).reshape(1,-1)
        current_load = np.array((self.loads[1][self.n],self.loads[2][self.n],self.loads[3][self.n])).reshape(1,-1)*(0.7+0.35*np.random.rand(1,3))
        init_state = np.concatenate((prev_flows,current_load),axis=1)
        self.state = init_state
        #self.prev_cost = np.array(0).reshape(1,-1)
        return self.state 

    def define_model(self):
        #print('make concrete model')
        self.model = m = ConcreteModel()
        #print('# make bus vars')
        m.ang = Var(self.buses,initialize=0.0)
        #print('# add slack angle constraint')
        m.slackang_cons = Constraint(expr=m.ang[0] == 0)
        flowvars = list(B.keys())
        flowvars += [(t,f) for (f,t) in flowvars]
        m.flowvars = Var(flowvars,domain = Reals)
        m.Ps = Var([0],domain = Reals)
        m.Pdg = Param([1,2],domain = Reals,initialize = 0,mutable = True)
        m.Load = Param([1,2,3],domain = Reals,initialize = 0,mutable = True)
        m.flowvars_cons = ConstraintList()
        m.flowvars_cons.add(expr = m.flowvars[0,1] == self.B[(0,1)]*(m.ang[0] - m.ang[1]))
        m.flowvars_cons.add(expr = m.flowvars[1,0] == self.B[(0,1)]*(m.ang[1] - m.ang[0]))
        m.flowvars_cons.add(expr = m.flowvars[1,2] == self.B[(1,2)]*(m.ang[1] - m.ang[2]))
        m.flowvars_cons.add(expr = m.flowvars[2,1] == self.B[(1,2)]*(m.ang[2] - m.ang[1]))
        m.flowvars_cons.add(expr = m.flowvars[2,3] == self.B[(2,3)]*(m.ang[2] - m.ang[3]))
        m.flowvars_cons.add(expr = m.flowvars[3,2] == self.B[(2,3)]*(m.ang[3] - m.ang[2]))
        m.flowvars_cons.add(expr = m.flowvars[3,0] == self.B[(0,3)]*(m.ang[3] - m.ang[0]))
        m.flowvars_cons.add(expr = m.flowvars[0,3] == self.B[(0,3)]*(m.ang[0] - m.ang[3]))
        m.powerbalance_cons = ConstraintList()
        m.powerbalance_cons.add(expr = m.Ps[0]-0 == m.flowvars[0,1]+m.flowvars[0,3])
        m.powerbalance_cons.add(expr = m.Pdg[1]-m.Load[1] == m.flowvars[1,0]+m.flowvars[1,2])
        m.powerbalance_cons.add(expr = m.Pdg[2]-m.Load[2] == m.flowvars[2,1]+m.flowvars[2,3])
        m.powerbalance_cons.add(expr = 0-m.Load[3] == m.flowvars[3,2]+m.flowvars[3,0])
        m.obj = Objective(expr = None,sense = minimize)

        # feasibility check model
        self.fmodel = fm = ConcreteModel()
        #print('# make bus vars')
        fm.ang = Var(self.buses,initialize=0.0)
        #print('# add slack angle constraint')
        fm.slackang_cons = Constraint(expr=fm.ang[0] == 0)
        flowvars = list(B.keys())
        flowvars += [(t,f) for (f,t) in flowvars]
        fm.flowvars = Var(flowvars,domain = Reals)
        fm.Ps = Var([0],domain = Reals)
        fm.Pdg = Var([1,2],domain = Reals,bounds = (0,50/100))
        fm.Load = Param([1,2,3],domain = Reals,initialize = 0,mutable = True)
        fm.flowvars_cons = ConstraintList()
        fm.flowvars_cons.add(expr = fm.flowvars[0,1] == self.B[(0,1)]*(fm.ang[0] - fm.ang[1]))
        fm.flowvars_cons.add(expr = fm.flowvars[1,0] == self.B[(0,1)]*(fm.ang[1] - fm.ang[0]))
        fm.flowvars_cons.add(expr = fm.flowvars[1,2] == self.B[(1,2)]*(fm.ang[1] - fm.ang[2]))
        fm.flowvars_cons.add(expr = fm.flowvars[2,1] == self.B[(1,2)]*(fm.ang[2] - fm.ang[1]))
        fm.flowvars_cons.add(expr = fm.flowvars[2,3] == self.B[(2,3)]*(fm.ang[2] - fm.ang[3]))
        fm.flowvars_cons.add(expr = fm.flowvars[3,2] == self.B[(2,3)]*(fm.ang[3] - fm.ang[2]))
        fm.flowvars_cons.add(expr = fm.flowvars[3,0] == self.B[(0,3)]*(fm.ang[3] - fm.ang[0]))
        fm.flowvars_cons.add(expr = fm.flowvars[0,3] == self.B[(0,3)]*(fm.ang[0] - fm.ang[3]))
        fm.powerbalance_cons = ConstraintList()
        fm.powerbalance_cons.add(expr = fm.Ps[0]-0 == fm.flowvars[0,1]+fm.flowvars[0,3])
        fm.powerbalance_cons.add(expr = fm.Pdg[1]-fm.Load[1] == fm.flowvars[1,0]+fm.flowvars[1,2])
        fm.powerbalance_cons.add(expr = fm.Pdg[2]-fm.Load[2] == fm.flowvars[2,1]+fm.flowvars[2,3])
        fm.powerbalance_cons.add(expr = 0-fm.Load[3] == fm.flowvars[3,2]+fm.flowvars[3,0])
        fm.flowvars_lims = ConstraintList()
        for (fb,tb) in self.lims.keys():
            fm.flowvars_lims.add(expr = fm.flowvars[fb,tb]<=self.lims[(fb,tb)])
            fm.flowvars_lims.add(expr = fm.flowvars[tb,fb]<=self.lims[(fb,tb)])
        fm.obj = Objective(expr = None,sense = minimize)
        self.solver = SolverFactory('ipopt')
    
    def get_reward(self):
        viols = [1 if abs(self.model.flowvars[k]()) > self.lims[k] else 0 for k in self.lims.keys()]
        viols += [1 if self.model.Ps[0]()<0 else 0]
        reward = (self.model.Pdg[1]()+self.model.Pdg[2]())/(self.model.Load[1]()+self.model.Load[2]()+self.model.Load[3]())*1
        #reward += sum([np.log(abs(1-abs(self.model.flowvars[k]())/self.lims[k]))/20 for k in self.lims.keys()])/4
        reward = -1 if sum(viols) else reward
        return reward
    
    def get_cost(self):
        cost = 0
        #cost = []
        for k in self.lims.keys():
            #cost.append(1.0 if np.abs(self.model.flowvars[k]())/self.lims[k]<=1.0 else -np.abs(self.model.flowvars[k]())/self.lims[k])
            #cost += np.abs(self.model.flowvars[k]())/self.lims[k] if np.abs(self.model.flowvars[k]())/self.lims[k]<=1.0 else np.abs(self.model.flowvars[k]())/self.lims[k]*4
            cost += 1 if np.abs(self.model.flowvars[k]())/self.lims[k]<=1.0 else 0
            #cost.append(np.abs(self.model.flowvars[k]())/self.lims[k])
        #cost = -(cost/4) if cost/4>1.0 else 1.0 
        cost = np.array(-1.0) if cost<4 else np.array(1.0)
        #cost = np.array(cost)
        return cost
    
    def step(self, action: np.ndarray,custom=False):
        # set dg and loads for this step
        #print(self.n)
        self.model.Pdg[1].value = action[0,0]
        self.model.Pdg[2].value = action[0,1]
        self.model.Load[1].value = self.state[0,-3] #self.loads[1][self.n]
        self.model.Load[2].value = self.state[0,-2] #self.loads[2][self.n]
        self.model.Load[3].value = self.state[0,-1] #self.loads[3][self.n]
        self.solver.solve(self.model)
        linefows = np.array([np.abs(self.model.flowvars[k]()) for k in self.B.keys()]).reshape(1,-1)
        next_loads = np.array((self.loads[1][self.n+1],self.loads[2][self.n+1],self.loads[3][self.n+1])).reshape(1,-1)
        next_loads_1 = np.random.normal(next_loads,0.2*next_loads).clip(0,next_loads)
        next_loads_2 = np.random.normal(next_loads,0.2*next_loads).clip(0,next_loads)
        next_state_alt = np.concatenate((linefows,next_loads_1),axis=1)
        reward = np.array(self.get_reward()).reshape(1,-1)
        cost = np.array(self.get_cost()).reshape(1,-1)
        done = np.array(False).reshape(1,-1)
        if (self.n==self.N-2):
            self.terminate = True
        self.n += 1
        if custom:
            self.state = np.concatenate((linefows,next_loads_2),axis=1)
        else:
            self.state = np.concatenate((linefows,next_loads),axis=1)
        exp = self.state,next_state_alt,reward,cost,done,self.terminate
        #self.prev_cost = cost
        return exp
    
    def step_new(self, action: np.ndarray):
        # set dg and loads for this step
        # first solve for the current state and then for the uncertain predicted
        # next state. The total reward is 0.8*reward(s)+0.2*I.cost(s+1)*reward(s')
        # where I is the indicator function. If the cost for next predicted state under
        # current action is -1, the total reward is just current reward. 
        self.model.Pdg[1].value = action[0,0]
        self.model.Pdg[2].value = action[0,1]
        self.model.Load[1].value = self.state[0,-3] #self.loads[1][self.n]
        self.model.Load[2].value = self.state[0,-2] #self.loads[2][self.n]
        self.model.Load[3].value = self.state[0,-1] #self.loads[3][self.n]
        self.solver.solve(self.model)
        linefows = np.array([np.abs(self.model.flowvars[k]()) for k in self.B.keys()]).reshape(1,-1)
        next_loads = np.array((self.loads[1][self.n+1],self.loads[2][self.n+1],self.loads[3][self.n+1])).reshape(1,-1)
        next_loads_1 = np.random.normal(next_loads,0.2*next_loads).clip(0,next_loads)
        next_loads_2 = np.random.normal(next_loads,0.2*next_loads).clip(0,next_loads)
        next_state_alt = np.concatenate((linefows,next_loads_1),axis=1)
        self.state = np.concatenate((linefows,next_loads_2),axis=1)
        reward = np.array(self.get_reward()).reshape(1,-1)
        cost = np.array(self.get_cost()).reshape(1,-1)
        done = np.array(False).reshape(1,-1)
        self.model.Pdg[1].value = action[0,0]
        self.model.Pdg[2].value = action[0,1]
        self.model.Load[1].value = next_state_alt[0,-3] #self.loads[1][self.n]
        self.model.Load[2].value = next_state_alt[0,-2] #self.loads[2][self.n]
        self.model.Load[3].value = next_state_alt[0,-1] #self.loads[3][self.n]
        self.solver.solve(self.model)
        reward_next = np.array(self.get_reward()).reshape(1,-1)
        cost_next = np.array(self.get_cost()).reshape(1,-1)
        reward_total = 0
        if cost ==-1:
            reward_total = reward
        elif cost_next ==-1:
            reward_total = reward
        else:
            reward_total = 0.5*reward+0.5*reward_next
        if (self.n==self.N-2):
            self.terminate = True
        self.n += 1
        exp = self.state,next_state_alt,reward_total,cost,done,self.terminate
        #self.prev_cost = cost
        return exp
    
    def step_test(self, action: np.ndarray):
        # set dg and loads for this step
        #print(self.n)
        self.model.Pdg[1].value = action[0,0]
        self.model.Pdg[2].value = action[0,1]
        self.model.Load[1].value = self.state[0,-3] #self.loads[1][self.n]
        self.model.Load[2].value = self.state[0,-2] #self.loads[2][self.n]
        self.model.Load[3].value = self.state[0,-1] #self.loads[3][self.n]
        self.solver.solve(self.model)
        linefows = np.array([np.abs(self.model.flowvars[k]()) for k in self.B.keys()]).reshape(1,-1)
        next_loads = np.array((self.loads[1][self.n+1],self.loads[2][self.n+1],self.loads[3][self.n+1])).reshape(1,-1)*(0.7+0.35*np.random.rand(1,3))
        self.state = np.concatenate((linefows,next_loads),axis=1)
        reward = np.array(self.get_reward()).reshape(1,-1)
        cost = np.array(self.get_cost()).reshape(1,-1)
        done = np.array(False).reshape(1,-1)
        if (self.n==self.N-2):
            self.terminate = True
        self.n += 1
        exp = self.state,reward,cost,done,self.terminate
        #self.prev_cost = cost
        return exp
    
    def cost_at_state(self, state:np.ndarray,action: np.ndarray):
        # set dg and loads for this step
        #print(self.n)
        loads = state[:,-3:].flatten()
        self.model.Pdg[1].value = action[0,0]
        self.model.Pdg[2].value = action[0,1]
        self.model.Load[1].value = loads[0]
        self.model.Load[2].value = loads[1]
        self.model.Load[3].value = loads[2]
        self.solver.solve(self.model)
        cost = self.get_cost()
        reward = self.get_reward()
        return cost,reward

    def run_loadflow(self):
        n = 0
        lineflows = {}
        while n < self.N:
            # set dg power to random value
            self.model.Pdg[1].value = self.gen_lims[1]*np.random.rand()
            self.model.Pdg[2].value = self.gen_lims[2]*np.random.rand()
            self.model.Load[1].value = self.loads[1][n]
            self.model.Load[2].value = self.loads[2][n]
            self.model.Load[3].value = self.loads[3][n]
            self.solver.solve(self.model)
            lineflows[n] = {k:np.abs(self.model.flowvars[k]()) for k in self.B.keys()}
            lineflows[n]['load'] = (self.model.Load[1](),self.model.Load[2](),self.model.Load[3]())
            lineflows[n]['gen'] = (self.model.Pdg[1](),self.model.Pdg[2](),self.model.Ps[0]())
            lineflows[n]['reward'] = self.get_reward()
            lineflows[n]['cost'] = self.get_cost()
            n += 1
        return lineflows

    def run_loadflow_f(self):
        n = 0
        lineflows = {}
        errors = {}
        while n < self.N:
            
            self.fmodel.Load[1].value = self.loads[1][n]
            self.fmodel.Load[2].value = self.loads[2][n]
            self.fmodel.Load[3].value = self.loads[3][n]
            info = self.solver.solve(self.fmodel)
            if (info.solver.termination_condition != TerminationCondition.optimal):
                errors[n] = (self.fmodel.Load[1](),self.fmodel.Load[2](),self.fmodel.Load[3]())
            else:
                lineflows[n] = {k:np.abs(self.fmodel.flowvars[k]()) for k in self.B.keys()}
                lineflows[n]['load'] = (self.fmodel.Load[1](),self.fmodel.Load[2](),self.fmodel.Load[3]())
                lineflows[n]['gen'] = (self.fmodel.Pdg[1](),self.fmodel.Pdg[2](),self.fmodel.Ps[0]())

            n += 1
        return lineflows,errors


