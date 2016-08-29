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
        self.sampled_episodes= []
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
            ret_batch = np.ones_like(cum_ret_batch) * cum_ret_batch[-1]
            
            self.sampled_episodes.append((obs_batch,act_batch,cum_ret_batch))
            self.curr_traj= {'obs':[],'act':[],'rew':[],'scc':[]}
            
    def get_batch(self):
        batch= []
        if len(self.sampled_episodes) >= self.batch_size:
            batch= self.sampled_episodes
            self.sampled_episodes= []
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
        
        with tf.variable_scope('encoder'):
            h = mlp(self.o_input, hidden_spec)
            
        with tf.variable_scope('policy'):
            w = tf.get_variable('w_out',shape=(hidden_spec[-1],self.a_dim),initializer=xavier())
            b = tf.get_variable('b_out',shape=(self.a_dim,))
            logits = tf.nn.xw_plus_b(h,w,b)
            
        self.logits = logits
        self.pi = tf.nn.softmax(logits)
        self.actiondist = tf.multinomial(logits, 1)
        
        #self.obj = tf.reduce_mean(tf.reduce_sum(logits - tf.expand_dims(tf.reduce_mean(logits,1),1),1)) * self.r_input
        #self.obj = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,self.a_input) * self.r_input
        #self.obj = tf.reduce_mean(tf.reduce_sum(logits - tf.expand_dims(tf.reduce_mean(logits,1),1),1)) * self.r_input
        #self.obj = tf.reduce_mean((gather_nd(logits, self.a_input) - tf.reduce_mean(logits,1)) * self.r_input)
        #self.obj = tf.reduce_mean(
            #tf.nn.sparse_softmax_cross_entropy_with_logits(logits,self.a_input) * self.r_input
            #)
        #self.obj = -tf.reduce_mean(
            #(tf.reduce_max(logits,1) - tf.reduce_mean(logits,1)) * self.r_input
            #)
        
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
            #[(obs, act, ret),...]
            sampled_episodes = batch
            grads_and_var= self.opt.compute_gradients(self.logits)
            grads_and_vars2=self.opt.compute_gradients(tf.reduce_mean(self.logits))
            
            weight_grad_est= {var:[] for _, var in grads_and_var}            
            for obs_batch, act_batch, ret_batch in sampled_episodes:
                for (grad, var),(grad2,var2) in zip(grads_and_var,grads_and_vars2):
                    gradient, gradient2, logits = self.sess.run([grad,grad2,self.logits],feed_dict={self.o_input:obs_batch,
                                                             self.a_input:act_batch[:,0],
                                                              #self.a_input:np.column_stack((np.arange(0,act_batch.shape[0]),
                                                                                            #act_batch[:,0])),
                                                              self.r_input:ret_batch[:,0]})
                    assert not np.isnan(gradient.sum())
                    #assert not (gradient.sum()==0.)
                    weight_grad_est[var].append(gradient)
                    
            for var, m_gradients in weight_grad_est.iteritems():
                gradient = np.mean(np.concatenate([g[...,None] for g in weight_grad_est[var]],
                                                  axis= -1),axis= -1)
                
                assert not np.isnan(gradient.sum())
                assign_op = var.assign_add(gradient)
                self.sess.run(assign_op)
            
        
            
        