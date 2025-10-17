from agenttraining import *


sigma = 0.2
epochs = 30
batch_size = 256
freq = 40
fillsize = 256
units=32

ts = time.time()
agent_ddpg = Agent(epochs=epochs,freq=freq,batch_size=batch_size,fillsize=fillsize,sigma=sigma,lr_pi=5e-4,lr_critic=1e-3,tau=0.001,units=units,fill_custom=True)
agent_ddpg.train_ddpg()
print(f'training time: {time.time()-ts}')


plt.figure()
plt.plot(np.array(agent_ddpg.episodic_violations)*100/400,'b--',label='DDPG')
plt.title(f'Total violations\n DDPG:{sum(agent_ddpg.episodic_violations)}')
plt.xlabel('episodes')
plt.ylabel('% violations/episode')
plt.legend(loc='lower right')
plt.figure()
plt.title('Episode Returns')
plt.plot(agent_ddpg.episodic_returns,'b--',label='DDPG')
plt.xlabel('episodes')
plt.ylabel('Returns/episode')
plt.legend(loc='lower right')
plt.figure()
plt.title(f'Test Episode Returns')
x = np.array(range(len(agent_ddpg.episodic_violations_test)))*2000/400
plt.plot(x,agent_ddpg.episodic_returns_test,'b--',label='DDPG')
plt.xlabel('episodes')
plt.ylabel('Returns/episode')
plt.legend(loc='lower right')
plt.show()
plt.figure()
plt.plot(x,np.array(agent_ddpg.episodic_violations_test)*100/400,'b--',label='DDPG')
plt.title('Test Episode Violations')
plt.xlabel('episodes')
plt.ylabel('% violations/episode')
plt.legend(loc='lower right')
plt.show()
