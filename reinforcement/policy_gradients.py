import abc
import numpy as np
import tensorflow as tf

from tensorflow.contrib.layers import xavier_initializer as xavier
from gym.spaces import Discrete, Box

import matplotlib.pyplot as plt
import gym
#import tf.contrib.distributions as dist

f = tf.nn.tanh
matmul = tf.matmul

def mlp(x, hidden_spec):
    layer = x
    h_prev= x.get_shape()[-1]
    for i, h in enumerate(hidden_spec):
        w= tf.get_variable('w%i'%i, shape=(h_prev,h),
                        initializer=xavier())
        b= tf.get_variable('b%i'%i, shape= (h,))
        layer= f(tf.nn.xw_plus_b(layer, w, b))
        h_prev = h
        
    return layer, h_prev

def get_dim(space):
    if type(space) is Box:
        dim = np.prod(space.shape)
    else:
        dim = space.n
    return dim

class Policy(object):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def __init__(self, env, B, T, hidden_spec):
        return    
    @abc.abstractmethod
    def fit(self, epochs, render):
        return
    @abc.abstractmethod
    def act(self, o):
        return
    
class Advantage(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def fit(self):
        pass
    @abc.abstractmethod
    def predict(self):
        pass
    
class Baseline(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def fit(self):
        pass
    @abc.abstractmethod
    def predict(self):
        pass
    
class SimpleBaseline(Baseline):
    """
    Computes the moving average of the expected return
    for any episode. (independent of observation).
    """
    def __init__(self):
        self.b = 0.
        self.c = 0.
    
    def fit(self, rew_B_T):
        c_new = np.prod(rew_B_T.shape)
        c_tot = self.c + c_new
        self.b = (self.c / c_tot)*self.b + (1. / c_tot)*rew_B_T.sum()
        self.c += c_new
    
    def predict(self, obs_B_T_Do):
        return self.b
        

class SimpleAdvantage(Advantage):
    """
    Cumulative sum 
    """
    def __init__(self, discount):
        self.discount = discount
        self.baseline = SimpleBaseline()
    
    def fit(self, tbatch):
        obs_B_T_Do, act_B_T_Da, rew_B_T = tbatch.unpack()
        self.baseline.fit(rew_B_T)
        
        halt= True
        
    def predict(self, tbatch):
        obs_B_T_Do, _, rew_B_T = tbatch.unpack()
        B, T = rew_B_T.shape
        
        discounts = np.ones(T,) * self.discount
        A_B_T = np.cumsum(rew_B_T,axis=1) \
            * (discounts ** np.arange(0,T)) \
            - self.baseline.predict(obs_B_T_Do)
        return A_B_T
    

class TrajBatch(object):
    def __init__(self,obs_B_T_Do,act_B_T_Da,rew_B_T):
        self.obs_B_T_Do=obs_B_T_Do
        self.act_B_T_Da=act_B_T_Da
        self.rew_B_T = rew_B_T
    
    def unpack(self):
        return self.obs_B_T_Do, self.act_B_T_Da, self.rew_B_T
    
def sample_trajs(policy,env,episodes=None,timesteps=None,render=False):
    if episodes is None:
        episodes =policy.B
    if timesteps is None:
        timesteps=policy.T
    
    obs_B_T, act_B_T, r_B_T = [], [], []
    for episode in range(episodes):
        o= env.reset()
        obs_T, act_T, r_T = [], [], []
        for t in range(timesteps):
            if render:
                env.render()
            
            a = policy.act(o)
            o_, r, done, info = env.step(a)
            
            if t == timesteps - 1:
                done= True
            
            o = o_.copy()
            
            obs_T.append(o)
            act_T.append(a)
            r_T.append(r)
            
            if done:
                break
            
        obs_B_T.append(
            np.array(obs_T)
        )
        act_B_T.append(
            np.array(act_T)
        )
        r_B_T.append(
            np.array(r_T)
        )
    obs_B_T_Do=np.array(obs_B_T)
    act_B_T_Da=np.array(act_B_T)
    rew_B_T= np.array(r_B_T)
    
    return TrajBatch(obs_B_T_Do, act_B_T_Da, rew_B_T)
            
        
class RandomPolicy(Policy):
    def __init__(self, env, adv, B, T, hidden_spec):
        #environment variables
        self.env = env
        self.B, self.T = B, T
        self.Do = Do = get_dim(env.observation_space)
        self.Da = Da = get_dim(env.action_space)
        
    def act(self, o):
        action = self.env.action_space.sample()
        return action
    
    def fit(self, epochs, render):
        raise NotImplementedError

class GaussianPolicy(Policy):
    def __init__(self, env, adv, B, T, hidden_spec, normalize= True):
        self.sess = tf.Session()
        
        #environment variables
        self.env = env
        self.adv = adv
        
        self.B, self.T = B, T
        self.Do = Do = get_dim(env.observation_space)
        self.Da = Da = get_dim(env.action_space)
        
        #Inputs
        self.o_input = tf.placeholder(tf.float32,shape=(None,Do))
        self.a_input = tf.placeholder(tf.float32,shape=(None,Da))
        self.adv_input = tf.placeholder(tf.float32,shape=(B,T))
        self.std_input = tf.placeholder(tf.float32,(None,Da))
        
        #Apply batch normalization on first layer.
        if normalize:
            x = tf.contrib.layers.batch_norm(self.o_input,
                                             trainable= False)
        else:
            x = tf.o_input        
                
        #Architecture
        with tf.variable_scope('encoder'):
            h, Dh = mlp(x, hidden_spec)
            
        with tf.variable_scope('policy'):
            w = tf.get_variable('w_out',shape=(Dh,Da),initializer=xavier())
            b = tf.get_variable('b_out',shape=(Da,))
            self.mu = mu = tf.nn.xw_plus_b(h,w,b)
            
            #self.logstd = tf.get_variable('logstd', shape=(2,))
            #self.mu, self.logstd = tf.split(1,2,self.logits)
            
        #Probability
        #self.std = tf.maximum(tf.exp(self.logstd),1.)
        self.dist = \
            tf.contrib.distributions.MultivariateNormalDiag(self.mu, self.std_input)
        
        #Training
        LL = self.dist.log_pdf(self.a_input)
        LL_B_T = tf.reshape(LL, (self.B,self.T))
        A_B_T = self.adv_input
        
        self.obj = tf.reduce_mean(
            tf.reduce_sum(LL_B_T * A_B_T,1)
            )
        
        #gradients
        self.params_and_grads= {}
        for param in tf.trainable_variables():
            grad = tf.gradients(self.obj, param)
            assert len(grad) == 1
            self.params_and_grads[param] = grad[0]
            
        self.sess.run(tf.initialize_all_variables())
    
    def act(self, o):
        if o.ndim == 1:
            o = o[None,...]
        elif o.ndim == 3:
            o = np.reshape(a, (self.B*self.T,self.Do))
        assert o.ndim == 2
        
        std = np.ones((1,1)) * 0.65
        feed = {self.o_input: o,
                self.std_input: std}
        mu,  = self.sess.run(self.mu, 
                               feed_dict= feed)
        
        action = np.random.normal(mu,std)
        assert not np.isnan(action)
        return np.array([action])
    
    def fit(self, epochs, lr=1e-10, render= False):
        avg_return= []
        for epoch in range(epochs):
            print "epoch: %i of %i"%(epoch,epochs)
            tbatch = sample_trajs(self,env)
            
            #Update advantage
            self.adv.fit(tbatch)            

            obs_B_T_Do, act_B_T_Da, rew_B_T = tbatch.unpack()
            adv_B_T = self.adv.predict(tbatch)
                        
            #Collapse timesteps into batch size:
            obs_B_T = np.reshape(obs_B_T_Do, (-1,obs_B_T_Do.shape[-1]))
            act_B_T = np.reshape(act_B_T_Da, (-1,act_B_T_Da.shape[-1]))
            
            #Update policy
            feed = {self.o_input: obs_B_T,
                    self.a_input: act_B_T,
                    self.adv_input: adv_B_T,
                    self.std_input: np.ones_like(act_B_T) * 0.65}
            
            ops= []
            for param, grad in self.params_and_grads.items():
                param_ = param.assign_add(lr * grad)
                ops.append( param.assign(param_) )
                
            #Report results
            _, grads = self.sess.run([ops,self.params_and_grads.values()],feed)
            for grad in grads:
                print 'gradient magnitude: %f'%(np.linalg.norm(grad))
            avg_return.append(np.mean(rew_B_T.sum(axis=1)))                
            print "Average Return: %f"%avg_return[-1]
            
            if render:
                sample_trajs(self,env,episodes=1,
                             render= True)
        plt.plot(avg_return)
        plt.show()
        
        halt= True
                
env = gym.make('Pendulum-v0')
adv = SimpleAdvantage(0.97)
policy = GaussianPolicy(env, adv, B=350,T=500,hidden_spec=[15,5])
policy.fit(1000, lr= 1e-6, render=True)
            
    
#class TrajBatch(object):
    #def __init__(self, batch_size):
        #self.batch_size = batch_size
        ##self.sampled_episodes= []
        #self.O, self.A, self.R, self.L = [], [], [], []
        #self.curr_traj= {'obs':[],'act':[],'rew':[],'scc':[]}
    
    #def update(self,o,a,r,o_,done):
        #self.curr_traj['obs'].append(o)
        #self.curr_traj['act'].append(a)        
        #self.curr_traj['rew'].append(r)
        #self.curr_traj['scc'].append(o_)        
        
        #if done:
            #obs_batch = np.concatenate([o[None] for o in self.curr_traj['obs']], axis= 0)
            #scc_batch = np.concatenate([s[None] for s in self.curr_traj['scc']], axis= 0)            
            #act_batch = np.concatenate([a[None] for a in self.curr_traj['act']], axis= 0)
            
            #cum_ret_batch = np.cumsum(self.curr_traj['rew'])[...,None]
            #ret_batch = cum_ret_batch - np.mean(cum_ret_batch)
            
            #self.O.append(obs_batch)
            #self.A.append(act_batch)
            #self.R.append(ret_batch)
            #self.L.append(len(self.curr_traj['obs']))
            
            ##self.sampled_episodes.append((obs_batch,act_batch,cum_ret_batch))
            #self.curr_traj= {'obs':[],'act':[],'rew':[],'scc':[]}
            
    #def get_batch(self):
        #batch= []
        #if len(self.O) >= self.batch_size:
            ##batch= self.sampled_episodes
            #batch = (np.concatenate(self.O),
                     #np.concatenate(self.A),
                     #np.concatenate(self.R),
                     #np.array(self.L))
            #self.O, self.A, self.R, self.L = [],[],[],[]
        #return batch
        
    
#class SoftmaxNet(Policy):
    #def __init__(self, env, hidden_spec, batch_size):
        #self.sess = tf.Session()
        #self.t_batch = TrajBatch(batch_size)
        #self.o_dim = get_dim(env.observation_space)
        #self.a_dim = get_dim(env.action_space)
        
        #self.o_input = tf.placeholder(tf.float32, shape=(None,self.o_dim), name='o_input')
        #self.a_input = tf.placeholder(tf.int32, shape=(None,), name='a_input')
        #self.r_input = tf.placeholder(tf.float32,shape=(None,),name='r_input')
        #self.len_input=tf.placeholder(tf.int32,shape=(None,),name='len_input')
        
        #with tf.variable_scope('encoder'):
            #h = mlp(self.o_input, hidden_spec)
            
        #with tf.variable_scope('policy'):
            #w = tf.get_variable('w_out',shape=(hidden_spec[-1],self.a_dim),initializer=xavier())
            #b = tf.get_variable('b_out',shape=(self.a_dim,))
            #logits = tf.nn.xw_plus_b(h,w,b)
            
        #self.logits = logits
        #self.pi = tf.nn.softmax(logits)
        #self.actiondist = tf.multinomial(logits, 1)
        
        ### ??
        #self.obj = tf.reduce_mean(
                #-1 * tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits,self.a_input)\
                #* self.r_input,
            #reduction_indices= 0
            #)
        
        #self.opt = tf.train.GradientDescentOptimizer(learning_rate=.1)
        #self.sess.run(tf.initialize_all_variables())
        
    #def act(self, o):
        #if o.ndim == 1:
            #o = o[None,...]
        #feed= {self.o_input: o}
        #action, logits= self.sess.run([self.actiondist,self.logits], feed_dict= feed)
        #return action[:,0]
    
    #def fit(self, o, a, r, o_, done):
        #self.t_batch.update(o,a,r,o_,done)
        #batch = self.t_batch.get_batch()
        #if batch != []:
            ##Stack of all episodes, all timesteps
            #(O,A,R,L) = batch
            
            #grads = tf.gradients(self.obj, tf.trainable_variables())
            #for var, grad in zip(tf.trainable_variables(),grads):
                #assign_op = var.assign_add(grad * R.mean())
                #self.sess.run(assign_op, feed_dict={self.o_input: O,
                                                    #self.a_input: A})
            
        