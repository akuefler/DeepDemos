import abc
import numpy as np
import tensorflow as tf

from tensorflow.contrib.layers import xavier_initializer as xavier
from gym.spaces import Discrete, Box

#import tf.contrib.distributions as dist

f = tf.nn.relu
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
        
    return layer

def get_dim(space):
    if type(space) is Box:
        dim = np.prod(space.shape)
    else:
        dim = space.n
    return dim

class Policy(object):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def __init__(self, env, hidden_spec):
        return    
    @abc.abstractmethod
    def fit(self, o, a, r, o_, done):
        return
    @abc.abstractmethod
    def act(self, o):
        return
    
class Critic(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def fit(self):
        pass
    @abc.abstractmethod
    def predict(self):
        pass
    
class TrajBatch(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        #self.sampled_episodes= []
        self.O, self.A, self.R, self.L = [], [], [], []
        self.curr_traj= {'obs':[],'act':[],'rew':[],'scc':[]}
    
    def update(self,o,a,r,o_,done):
        self.curr_traj['obs'].append(o)
        self.curr_traj['act'].append(a)        
        self.curr_traj['rew'].append(r)
        self.curr_traj['scc'].append(o_)        
        
        if done:
            obs_batch = np.concatenate([o[None] for o in self.curr_traj['obs']], axis= 0)
            scc_batch = np.concatenate([s[None] for s in self.curr_traj['scc']], axis= 0)            
            act_batch = np.concatenate([a[None] for a in self.curr_traj['act']], axis= 0)
            
            cum_ret_batch = np.cumsum(self.curr_traj['rew'])[...,None]
            ret_batch = cum_ret_batch - np.mean(cum_ret_batch)
            
            self.O.append(obs_batch)
            self.A.append(act_batch)
            self.R.append(ret_batch)
            self.L.append(len(self.curr_traj['obs']))
            
            #self.sampled_episodes.append((obs_batch,act_batch,cum_ret_batch))
            self.curr_traj= {'obs':[],'act':[],'rew':[],'scc':[]}
            
    def get_batch(self):
        batch= []
        if len(self.O) >= self.batch_size:
            #batch= self.sampled_episodes
            batch = (np.concatenate(self.O),
                     np.concatenate(self.A),
                     np.concatenate(self.R),
                     np.array(self.L))
            self.O, self.A, self.R, self.L = [],[],[],[]
        return batch
        
    
class SoftmaxNet(Policy):
    def __init__(self, env, hidden_spec, batch_size):
        self.sess = tf.Session()
        self.t_batch = TrajBatch(batch_size)
        self.o_dim = get_dim(env.observation_space)
        self.a_dim = get_dim(env.action_space)
        
        self.o_input = tf.placeholder(tf.float32, shape=(None,self.o_dim), name='o_input')
        self.a_input = tf.placeholder(tf.int32, shape=(None,), name='a_input')
        self.r_input = tf.placeholder(tf.float32,shape=(None,),name='r_input')
        self.len_input=tf.placeholder(tf.int32,shape=(None,),name='len_input')
        
        with tf.variable_scope('encoder'):
            h = mlp(self.o_input, hidden_spec)
            
        with tf.variable_scope('policy'):
            w = tf.get_variable('w_out',shape=(hidden_spec[-1],self.a_dim),initializer=xavier())
            b = tf.get_variable('b_out',shape=(self.a_dim,))
            logits = tf.nn.xw_plus_b(h,w,b)
            
        self.logits = logits
        self.pi = tf.nn.softmax(logits)
        self.actiondist = tf.multinomial(logits, 1)
        
        ## ??
        self.obj = tf.reduce_mean(
                -1 * tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits,self.a_input)\
                * self.r_input,
            reduction_indices= 0
            )
        
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=.1)
        self.sess.run(tf.initialize_all_variables())
        
    def act(self, o):
        if o.ndim == 1:
            o = o[None,...]
        feed= {self.o_input: o}
        action, logits= self.sess.run([self.actiondist,self.logits], feed_dict= feed)
        return action[:,0]
    
    def fit(self, o, a, r, o_, done):
        self.t_batch.update(o,a,r,o_,done)
        batch = self.t_batch.get_batch()
        if batch != []:
            #Stack of all episodes, all timesteps
            (O,A,R,L) = batch
            
            grads = tf.gradients(self.obj, tf.trainable_variables())
            for var, grad in zip(tf.trainable_variables(),grads):
                assign_op = var.assign_add(grad * R.mean())
                self.sess.run(assign_op, feed_dict={self.o_input: O,
                                                    self.a_input: A})
            
        