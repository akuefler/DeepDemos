import numpy as np
import tensorflow as tf
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import pyplot as plt

import data
from data.visualization import Visualizer
    
def flatten(t):
    flat_dim = np.prod(t.get_shape().as_list())
    return tf.reshape(t, (-1, flat_dim))

def kl_divergence(W_post):
    """
    Computes the (approximate) kl divergence between a multivariable, diagonal gaussian
    and standard normal gaussian.
    
    W_post (tensor): mean and rho values for posterior weight matrix, concatenated along axis 1.
    """
    mu, rho = tf.split(1,2,W_post)
    sig = tf.log(1.0 + tf.exp(rho)) # stdev must be positive.
    mu, sig = flatten(mu), flatten(sig)   
    
    posterior = tf.contrib.distributions.MultivariateNormalDiag(
        mu, sig, name="posterior")
    prior = tf.contrib.distributions.MultivariateNormalDiag(
        tf.zeros_like(mu),tf.ones_like(sig),name="prior")

    return posterior.log_pdf(posterior.sample()) - prior.log_pdf(posterior.sample())
  
class NNet():
    """
    implements two-layer neural networks in both bayesian and deterministic flavors.
    i_dim : input dimensionality
    h_dim : hidden layer dimensionality
    o_dim : output layer dimensionality
    f : activation function
    bayes : boolean flag to indicate weight uncertainty
    """
    def __init__(self, vis, i_dim, h_dim, o_dim, f= tf.nn.tanh, bayes= True,
                 batch_size= 10, learning_rate= 1e-2, reg= 0.0):
        self.sess = tf.Session()
        self.bayes= bayes
        self.vis = vis
        
        #Define batch sizes for true data and noise samples.
        self.data_bs = batch_size       
        
        with tf.variable_scope('inputs'):
            self.x_input=x= tf.placeholder(tf.float32, shape=(None,i_dim), name='x_input')
            self.y_input=y= tf.placeholder(tf.float32, shape=(None,o_dim), name='y_input')
        with tf.variable_scope('params'):
            if bayes:
                #used to weight KL-divergence (regularization) term during minibatch learning.
                self.complexity_weight= tf.placeholder(dtype=tf.float32, shape=(), name='c_weight')
                
                # tunable mu and std parameter weight matrices
                w1_param = tf.get_variable('w1', shape=(i_dim,h_dim * 2), dtype=tf.float32)
                w2_param = tf.get_variable('w2',shape= (h_dim,o_dim * 2),dtype=tf.float32)
                
                def reparam(t):
                    """
                    can't backprop through stochastic weights.
                    employs reparameterization trick.
                    """
                    mu, rho = tf.split(1,2,t)
                    sig = tf.log(1.0 + tf.exp(rho)) # stdev must be positive
                    return mu + sig * tf.random_normal(sig.get_shape())
                
                # stochastic weight matrices
                w1 = reparam(w1_param)
                w2 = reparam(w2_param)
                
            else:
                # deterministic weight matrices
                w1 = tf.get_variable('w1', shape=(i_dim,h_dim), dtype=tf.float32)
                w2 = tf.get_variable('w2',shape= (h_dim,o_dim),dtype=tf.float32)
                
            b1 = tf.get_variable('b1',shape=(h_dim,), dtype=tf.float32)
            b2 = tf.get_variable('b2',shape=(o_dim,), dtype=tf.float32)                
                
            self.params = {'w1':w1,'w2':w2,'b1':b1,'b2':b2}            
               
        with tf.variable_scope('training'):
            if self.bayes:
                c1 = kl_divergence(w1_param)
                c2 = kl_divergence(w2_param)
                complexity_term = self.complexity_weight * tf.reduce_sum([c1,c2])
            else:
                complexity_term = reg * tf.reduce_sum([tf.nn.l2_loss(w1),tf.nn.l2_loss(w2)])
                
            #Define network/model architecture
            self.model=y_pred= tf.matmul(f(tf.matmul(x,w1)+b1),w2)+b2
            
            #Maximum likelihood cost component (reduces to l2 norm for Gaussians)
            likelihood_term = tf.reduce_mean(tf.reduce_mean((y - y_pred) ** 2,reduction_indices= 0))
            
            #Objective function.
            self.cost = likelihood_term + complexity_term
            
            #TensorFlow optimizers
            self.adam= tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.opt = self.adam.minimize(self.cost)

        self.sess.run(tf.initialize_all_variables())
            
    def fit(self,X_t,Y_t,X_v,Y_v,epochs=100):
        """
        minibatch gradient descent.
        
        X_t, Y_t : training set and labels
        X_v, Y_v : validation set and labs
        epochs : number of epochs
        """
        N = X_t.shape[0]
        M = int(N/self.data_bs)
        
        feed_v = {self.x_input: X_v, self.y_input: Y_v}
        if self.bayes:
            feed_v.update({self.complexity_weight: 0.})
        
        for epoch in range(epochs):
            print "Epoch: %i"%epoch
            p = np.random.permutation(N)
            X_t= X_t[p]
            Y_t= Y_t[p]
            losses_t= []
            losses_v= []
            
            for i in range(0,N,self.data_bs):
                X_batch = X_t[i:i+self.data_bs,:]
                Y_batch = Y_t[i:i+self.data_bs,:]
                
                feed_t = {self.x_input: X_batch, self.y_input: Y_batch}
                if self.bayes:
                    pi = (2. ** (M-i))/(2.**M - 1.)
                    feed_t.update({self.complexity_weight:pi})
                
                loss_t, _ = self.sess.run([self.cost, self.opt],feed_t)
                losses_t.append(loss_t)
            
            loss_t = np.mean(losses_t)
            loss_v = self.sess.run(self.cost, feed_v)
            
            print "Train Loss: {} == Valid. Loss: {}".format(loss_t,loss_v)
            
            self.vis.display_loss(np.mean(losses_t), loss_v)
            
            Y_p, Y_std = self.predict(X_v)
            self.vis.display_data(epoch, X_v, Y_p, Y_std)
                
    def predict(self, X, n = 100):
        """
        Output predictions on input data X.
        
        if bayesian, sample n predictions and return mean and stdev.
        """
        if self.bayes:
            preds = []
            for _ in xrange(n):
                preds.append(self._predict(X))
                
            prediction = np.mean(preds, axis = 0)
            confidence = np.std(preds, axis = 0)
        else:
            prediction = self._predict(X)
            confidence = None
            
        return prediction, confidence
    
    def _predict(self, X):
        feed = {self.x_input:X}
        return self.sess.run(self.model, feed)
