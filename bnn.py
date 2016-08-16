import numpy as np
import tensorflow as tf
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import pyplot as plt

import data

"""
Works after about 5000 epochs.
"""

N = 1000; V= int(N * 0.7); M= 2; H= 5*M; O= 1
sig= 0.2; EPOCHS= 5000
X, Y = data.pringle(N,M,sig)
    
def gauss_ll(params, v_params= None):
    if v_params is None:
        mu= {name : tf.constant(np.zeros(tensor.get_shape()[1:]).astype('float32')) for name, tensor in params.items()}
        std= {name : tf.constant(np.ones(tensor.get_shape()[1:]).astype('float32')) for name, tensor in params.items()}
        v_params= {'mu':mu,'std':std}
    
    logprobs= []
    #For each network param (w1, b1, w2, b2) 
    for name, tensor in params.items():
        mu = tf.reshape(v_params['mu'][name], (np.prod(v_params['mu'][name].get_shape()).value,))
        std= tf.reshape(v_params['std'][name],(np.prod(v_params['std'][name].get_shape()).value,))
        gaussian = tf.contrib.distributions.MultivariateNormalDiag(mu, std)
        lps=[]
        #For each sample in the batch of "variational posteriors" weights. E~q(w|theta)
        for k in range(tensor.get_shape()[0].value):
            row= tensor[k,:,:]
            v= tf.reshape(row, (np.prod(row.get_shape()).value,))
            lp= gaussian.log_prob(v)
            
            lps.append(lp)
        logprobs.append(tf.reduce_mean(lps))
    
    return tf.reduce_sum(logprobs) ##Doesn't seem to matter...
    #return tf.reduce_mean(logprobs)
    
class Visualizer():
    def __init__(self, dim, xlim, ylim, epochs= None, interval= 2):
        self.f, self.axs = plt.subplots(nrows=1,ncols=1 + dim)
        self.interval= interval; self.epochs= epochs
        self.losses= []

        if self.epochs:
            self.axs[0].set_xlim((0,epochs))
        for ax in self.axs[1:]:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
    
    def display_loss(self,loss):
        plt.ion()
        plt.show()
        self.losses.append(loss)
        self.axs[0].cla()
        if self.epochs:
            self.axs[0].set_xlim((0,self.epochs))        
        self.axs[0].plot(self.losses,color='k')
        
        plt.draw()
        plt.pause(1e-10)        
                    
    def display_data(self,X,Y,iteration):
        if iteration % self.interval == 0:
            return
        plt.ion()
        plt.show()
        for i, ax in enumerate(self.axs[1:]):
            ax.cla()
            ax.autoscale(False)
            if Y.ndim == 2:
                ax.scatter(X[:,i],Y)
            else:
                Y_mu= Y.mean(axis=0)
                Y_std=Y.std(axis =0)
                ax.errorbar(X[:,i],Y_mu,yerr=Y_std,
                            fmt='o',ecolor='r')
            
        plt.draw()
        plt.pause(1e-10)
  
class NNet():
    def __init__(self, vis, i_dim, h_dim, o_dim, f= tf.nn.tanh, bayes= False):
        self.sess = tf.Session()
        self.bayes= bayes
        self.vis = vis
        
        #
        self.data_bs = 10
        self.weight_bs= 5        
        
        with tf.variable_scope('inputs'):
            self.x_input=x= tf.placeholder(tf.float32, shape=(None,i_dim), name='x_input')
            self.y_input=y= tf.placeholder(tf.float32, shape=(None,o_dim), name='y_input')
        with tf.variable_scope('params'):
            if bayes:
                #used to weight KL-divergence (regularization) term during minibatch learning.
                self.complexity_weight= tf.placeholder(dtype=tf.float32, shape=())
                
                #Define mu for each weight
                w1_mu = tf.get_variable('w1_mu', shape=(i_dim,h_dim), dtype=tf.float32)
                b1_mu = tf.get_variable('b1_mu',shape=(h_dim,), dtype=tf.float32)
                w2_mu = tf.get_variable('w2_mu',shape= (h_dim,o_dim),dtype=tf.float32)
                b2_mu = tf.get_variable('b2_mu',shape=(o_dim,), dtype=tf.float32)
                
                #Define "rho" for each weight
                w1_rho = tf.get_variable('w1_rho', shape=(i_dim,h_dim), dtype=tf.float32)
                b1_rho = tf.get_variable('b1_rho',shape=(h_dim,), dtype=tf.float32)            
                w2_rho = tf.get_variable('w2_rho',shape= (h_dim,o_dim),dtype=tf.float32)
                b2_rho = tf.get_variable('b2_rho',shape=(o_dim,), dtype=tf.float32)                
                
                #Define std for each weight (function of rho, ensuring stds are positive)
                w1_std = tf.add(1., tf.exp(w1_rho), name= 'w1_std')
                b1_std = tf.add(1., tf.exp(b1_rho), name= 'b1_std')
                w2_std = tf.add(1., tf.exp(w2_rho), name= 'w2_std')
                b2_std = tf.add(1., tf.exp(b2_rho), name= 'b2_std')                 
                
                #placeholders for parameter-free noise samples.
                w1_e = tf.placeholder(name='w1_e', shape=(self.weight_bs,i_dim,h_dim), dtype=tf.float32)
                b1_e = tf.placeholder(name='b1_e',shape=(self.weight_bs,1,h_dim,), dtype=tf.float32)
                w2_e = tf.placeholder(name='w2_e',shape= (self.weight_bs,h_dim,o_dim),dtype=tf.float32)
                b2_e = tf.placeholder(name='b2_e',shape=(self.weight_bs,1,o_dim,), dtype=tf.float32)
                self.samples = {'w1':w1_e,'w2':w2_e,'b1':b1_e,'b2':b2_e}
                
                #variational parameters (i.e., the trainable params in bayesian inference)
                self.v_params= {'mu':{'w1':w1_mu,'w2':w2_mu,'b1':b1_mu,'b2':b2_mu},
                                   'std':{'w1':w1_std,'w2':w2_std,'b1':b1_std,'b2':b2_std}}                
                
                #reparameterization trick! (scale noise with std, translate with mean)
                w1= tf.add(w1_mu, w1_std * w1_e,name='w1')
                b1= tf.add(b1_mu, b1_std * b1_e,name='b1')
                w2= tf.add(w2_mu, w2_std * w2_e,name='w2')
                b2= tf.add(b2_mu, b2_std * b2_e,name='b2')
            else:
                #define network weights
                w1 = tf.get_variable('w1', shape=(i_dim,h_dim), dtype=tf.float32)
                b1 = tf.get_variable('b1',shape=(h_dim,), dtype=tf.float32)
                w2 = tf.get_variable('w2',shape= (h_dim,o_dim),dtype=tf.float32)
                b2 = tf.get_variable('b2',shape=(o_dim,), dtype=tf.float32)                
                
            self.params = {'w1':w1,'w2':w2,'b1':b1,'b2':b2}            
               
        with tf.variable_scope('training'):
            if self.bayes:
                #Repeat x and y, because each epoch we train a different network on the data.
                x = tf.tile(tf.expand_dims(x,0), (self.weight_bs,1,1))
                y = tf.tile(tf.expand_dims(y,0), (self.weight_bs,1,1))
                matmul = tf.batch_matmul #Need batch matmul because both data AND noise come in batches.
                complexity_term = self.complexity_weight *\
                    (gauss_ll(self.params, v_params=self.v_params)-gauss_ll(self.params))
            else:
                matmul = tf.matmul
                complexity_term= 0.0
                
            #Define network/model architecture
            self.model=y_pred= matmul(f(matmul(x,w1)+b1),w2)+b2
            
            #Maximum likelihood cost component (reduces to l2 norm for Gaussians)
            likelihood_term = tf.reduce_mean(tf.reduce_sum((y - y_pred) ** 2,reduction_indices= 0))
            
            #Objective function.
            self.cost = likelihood_term + complexity_term
            
            #TensorFlow optimizers
            self.adam= tf.train.AdamOptimizer()
            self.opt = self.adam.minimize(self.cost)

        self.sess.run(tf.initialize_all_variables())
            
    def fit(self,X_t,Y_t,X_v,Y_v,epochs=100):
        N = X_t.shape[0]
        M = int(N/self.data_bs)        
        
        for epoch in range(epochs):
            print "Epoch: %i"%epoch
            p = np.random.permutation(N)
            X_t= X_t[p]
            Y_t= Y_t[p]
            losses= []
            for i in range(0,N,self.data_bs):
                X_batch= X_t[i:i+self.data_bs,:]
                Y_batch= Y_t[i:i+self.data_bs,:]
                feed= {self.x_input: X_batch, self.y_input: Y_batch}
                if self.bayes:
                    feed_samples= {tensor: np.random.normal(0,1,tensor.get_shape())\
                                   for _, tensor in self.samples.items()}
        
                    pi = (2. ** (M-i))/(2.**M - 1.)
                    feed.update(feed_samples)
                    feed.update({self.complexity_weight:pi})         
                
                loss, _ = self.sess.run([self.cost, self.opt],feed)
                losses.append(loss)
                
            print "Average Loss: %f"%np.mean(losses)
            Y_p = self.predict(X_v)
            vis.display_loss(np.mean(losses))
            vis.display_data(X_v, Y_p, epoch)
                
    
    def predict(self,X, n= 100):
        feed={self.x_input:X}
        if self.bayes:
            feed_samples= {tensor: np.random.normal(0,1,tensor.get_shape())\
                           for name, tensor in self.samples.items()}
            feed.update(feed_samples)
            predictions = self.sess.run(self.model, feed)
            
            #Retrieve mu and std
            mu= self.sess.run(self.v_params['mu'].values())
            std=self.sess.run(self.v_params['std'].values())
            
            return predictions
                        
        else:
            prediction= self.sess.run(self.model, feed)
                
            return prediction
    
X_t= X[:V]
Y_t= Y[:V]
X_v= X[V:]
Y_v= Y[V:]

vis = Visualizer(dim= 2, xlim=(X.min(),X.max()),ylim=(Y.min(),Y.max()), epochs= None, interval= 15)
nn = NNet(vis, M, H, O, bayes= True)
nn.fit(X_t, Y_t, X_v, Y_v, epochs= EPOCHS)

Y_p= nn.predict(X_v, n= 100)

#(X_t, Y_t, X_v, Y_p)