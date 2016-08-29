import gym
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from policy_gradients import SoftmaxNet
M = 10
EPSILON= 1e-2
EPISODES= 100 * M
TIMESTEPS= 150
BIN = 5
#ETA = 1e-1
ETA= 1e-1
GAMMA = 0.95
env = gym.make('CartPole-v0')

agent = SoftmaxNet(env, [10,10], batch_size= M)

for episode in range(EPISODES):
    print("Episode: %i of %i"%(episode,EPISODES))
    o= env.reset()
    current_reward= 0
    rewards= []
    for t in range(TIMESTEPS):
        env.render()
        
        a = agent.act(o)
        o_, r, done, info = env.step(a)
        
        if t == TIMESTEPS - 1:
            done= True
        
        agent.fit(o,a,r,o_,done)
        o = o_.copy()
        
        current_reward += r
        
        if done:
            break
    rewards.append(current_reward)
        
plt.plot(rewards)

plt.show()

halt= True