import numpy as np
import tensorflow as tf
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import pyplot as plt


"""
Works after about 5000 epochs.
"""

N = 1000; V= int(N * 0.7); M= 2; H= 5*M; O= 1
sig= 0.2

X = np.random.uniform(-5,5,(N,M))
Z_w1 = np.random.normal(0.,1., (N,M,H))
Z_b1 = np.random.normal(0.,1., (N,1,H))
Z_w2= np.random.normal(0.,1., (N,H,O))
Z_b2= np.random.normal(0.,1., (N,1,O))

#Z_w1 = np.zeros_like(Z_w1)
#Z_b1 = np.zeros_like(Z_b1)
#Z_w2 = np.zeros_like(Z_w2)
#Z_b2 = np.zeros_like(Z_b2)

#Y = (np.cos(X[:,0]) + np.sin(X[:,1]) + np.random.normal(0,sig,(N,)))[...,None]
Y = (X[:,0]**2 - X[:,1]**2 + np.random.normal(0,sig,(N,)))[...,None]


def display(X,Y,X_p,Y_p):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0],X[:,1],Y)
    ax.scatter(X_p[:,0],X_p[:,1],Y_p, c= 'r')
    plt.show()
    
    f, axs = plt.subplots(1,2)
    for i, ax in enumerate(axs):
        ax.scatter(X[:,i],Y)
        ax.scatter(X_p[:,i],Y_p, c= 'r')
    plt.show()
    
#def gauss_ll(params, v_params= None):
    #if v_params is None:
        #mu= {name : tf.constant(np.zeros(tensor.get_shape()).astype('float32')) for name, tensor in params.items()}
        #std= {name : tf.constant(np.ones(tensor.get_shape()).astype('float32')) for name, tensor in params.items()}
        #v_params= {'mu':mu,'std':std}
    
    #logprobs= []
    #for name, tensor in params.items():
        #mu = tf.reshape(v_params['mu'][name], (np.prod(v_params['mu'][name].get_shape()).value,))
        #std= tf.reshape(v_params['std'][name],(np.prod(v_params['std'][name].get_shape()).value,))
        #gaussian = tf.contrib.distributions.MultivariateNormalDiag(mu, std)
        #logprobs.append(gaussian.log_prob(tf.reshape(tensor, (np.prod(tensor.get_shape()).value,))))
        
    #return tf.reduce_sum(logprobs)
    
#def gauss_ll(w, mu, std):
    #if np.ndim(mu) == 0:
        #dim = np.prod(w.get_shape()).value
        #mu = tf.constant(np.zeros(dim,),dtype=tf.float32)
        #std= tf.constant(np.ones(dim,), dtype=tf.float32)
    #gaussian = tf.contrib.distributions.MultivariateNormalDiag(mu, std)
    #ll = gaussian.log_prob(tf.reshape(w, (np.prod(w.get_shape()).value,)))
    #return ll
    
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
    
    
class nnet():
    def __init__(self, i_dim, h_dim, o_dim, f= tf.nn.tanh, bayes= False):
        self.sess = tf.Session()
        self.bayes= bayes
        self.batch_size=e_batch= 10        
        
        with tf.variable_scope('inputs'):
            self.x_input=x= tf.placeholder(tf.float32, shape=(None,i_dim), name='x_input')
            self.y_input=y= tf.placeholder(tf.float32, shape=(None,o_dim), name='y_input')
        with tf.variable_scope('params'):
            if bayes:
                self.complexity_weight= tf.placeholder(dtype=tf.float32, shape=())
                
                #Define mu for each weight
                w1_mu = tf.get_variable('w1_mu', shape=(i_dim,h_dim), dtype=tf.float32)
                b1_mu = tf.get_variable('b1_mu',shape=(h_dim,), dtype=tf.float32)
                
                w2_mu = tf.get_variable('w2_mu',shape= (h_dim,o_dim),dtype=tf.float32)
                b2_mu = tf.get_variable('b2_mu',shape=(o_dim,), dtype=tf.float32)
                
                #Define rho for each weight
                w1_rho = tf.get_variable('w1_rho', shape=(i_dim,h_dim), dtype=tf.float32)
                b1_rho = tf.get_variable('b1_rho',shape=(h_dim,), dtype=tf.float32)
                
                w2_rho = tf.get_variable('w2_rho',shape= (h_dim,o_dim),dtype=tf.float32)
                b2_rho = tf.get_variable('b2_rho',shape=(o_dim,), dtype=tf.float32)
                
                #Define std for each weight
                w1_std = tf.add(1., tf.exp(w1_rho), name= 'w1_std')
                b1_std = tf.add(1., tf.exp(b1_rho), name= 'b1_std')
                
                w2_std = tf.add(1., tf.exp(w2_rho), name= 'w2_std')
                b2_std = tf.add(1., tf.exp(b2_rho), name= 'b2_std')                
                
                #Store variational params together
                self.v_params=vp= {'mu':{'w1':w1_mu,'w2':w2_mu,'b1':b1_mu,'b2':b2_mu},
                                'rho':{'w1':w1_rho,'w2':w2_rho,'b1':b1_rho,'b2':b2_rho},
                                'std':{'w1':w1_std,'w2':w2_std,'b1':b1_std,'b2':b2_std}}                
                
                #Arguments for creating the weights in the network.
                create_param= tf.placeholder
                kwargs= {}
                extract = lambda x : x.split('/')[-1].split(':')[0]
                reparam = lambda x : tf.add(vp['mu'][extract(x.name)], (vp['std'][extract(x.name)] * x))
                
            else:
                create_param= tf.get_variable
                kwargs= {'initializer':tf.contrib.layers.xavier_initializer()}
                reparam = lambda x : x
            
            w1 = create_param(name='w1', shape=(e_batch,i_dim,h_dim), dtype=tf.float32, **kwargs)
            b1 = create_param(name='b1',shape=(e_batch,1,h_dim,), dtype=tf.float32)
            
            w2 = create_param(name='w2',shape= (e_batch,h_dim,o_dim),dtype=tf.float32, **kwargs)
            b2 = create_param(name='b2',shape=(e_batch,1,o_dim,), dtype=tf.float32)
            
            #Reparameterize stochastic weights
            self.samples = {'w1':w1,'w2':w2,'b1':b1,'b2':b2}            
            #w1= reparam(w1); b1= reparam(b1); w2= reparam(w2); b2= reparam(b2)
            w1_r= tf.add(w1_mu, w1_std * w1,name='w1_r')
            b1_r= tf.add(b1_mu, b1_std * b1,name='b1_r')
            
            w2_r= tf.add(w2_mu, w2_std * w2,name='w2_r')
            b2_r= tf.add(b2_mu, b2_std * b2,name='b2_r')            
            
            self.params = {'w1':w1_r,'w2':w2_r,'b1':b1_r,'b2':b2_r}            
               
        with tf.variable_scope('training'):
            x = tf.tile(tf.expand_dims(x,0), (e_batch,1,1))
            y = tf.tile(tf.expand_dims(y,0), (e_batch,1,1))
            
            #self.model = y_hat = tf.matmul(f(tf.matmul(x,w1)+b1),w2)+b2
            #self.model = y_hat = tf.batch_matmul(f(tf.batch_matmul(x,w1)+b1),w2)+b2
            
            h1=f(tf.batch_matmul(x,w1_r)+b1_r)
            h2=tf.batch_matmul(h1,w2_r)+b2_r
            self.model = y_hat = h2
            
            #self.likely_J = self.J = tf.nn.l2_loss(y - y_hat)
            #self.likely_J = tf.reduce_sum(tf.reduce_sum((y - y_hat) ** 2))
            self.likely_J = tf.reduce_mean(tf.reduce_sum((y - y_hat) ** 2,reduction_indices= 0))
            if self.bayes:
                self.J = self.complexity_weight *\
                    (gauss_ll(self.params, v_params=self.v_params) - gauss_ll(self.params))\
                    + self.likely_J
            
            self.adam= tf.train.AdamOptimizer()
            self.opt = self.adam.minimize(self.J)

        self.sess.run(tf.initialize_all_variables())
            
    def fit(self,X,Y,Z_w1, Z_b1, Z_w2, Z_b2,epochs=100):
        N = X.shape[0]
        batch_size = self.batch_size
        M = int(N/batch_size)        
        
        for epoch in range(epochs):
            print "Epoch: %i"%epoch
            p = np.random.permutation(N)
            X= X[p]
            Y= Y[p]
            
            Z_w1= Z_w1[p]
            Z_b1= Z_b1[p]
            
            Z_w2= Z_w2[p]
            Z_b2= Z_b2[p]
            
            losses= []
            pi= np.random.randn(M);pi= np.exp(pi)/np.sum(np.exp(pi))
            for i in range(0,N,batch_size):
                X_batch= X[i:i+batch_size,:]
                Y_batch= Y[i:i+batch_size,:]
                feed= {self.x_input: X_batch, self.y_input: Y_batch}
                if self.bayes:
                    feed_samples= {tensor: np.random.normal(0,1,tensor.get_shape())\
                                   for name, tensor in self.samples.items()}
                    #feed_samples= {tensor: np.ones(tensor.get_shape())\
                                   #for name, tensor in self.samples.items()}
                    w1_batch = Z_w1[i:i+batch_size,:]
                    b1_batch = Z_b1[i:i+batch_size,:]
                    w2_batch = Z_w2[i:i+batch_size,:]
                    b2_batch = Z_b2[i:i+batch_size,:] 
                    
                    pi = (2. ** (M-i))/(2.**M - 1.)
                    #pi = 0.01
                    ##loss approx. 370
                    #feed_samples= {self.samples['w1']: w1_batch,
                                   #self.samples['b1']: b1_batch,
                                   #self.samples['w2']: w2_batch,
                                   #self.samples['b2']: b2_batch
                    #}
                    feed.update(feed_samples)
                    feed.update({self.complexity_weight:pi})
                    #loss, _ = self.sess.run([self.J, self.opt],feed)
                    #losses.append(loss)                        
                
                loss, _ = self.sess.run([self.J, self.opt],feed)
                losses.append(loss)
                
            print "Average Loss: %f"%np.mean(losses)
                
    
    def predict(self,X, n= 100):
        feed={self.x_input:X}
        if self.bayes:
            feed_samples= {tensor: np.random.normal(0,1,tensor.get_shape())\
                           for name, tensor in self.samples.items()}
            #feed_samples= {tensor: np.ones(tensor.get_shape())\
                                               #for name, tensor in self.samples.items()}
            #feed_samples= {tensor: np.zeros(tensor.get_shape())\
                                               #for name, tensor in self.samples.items()}
            
            feed.update(feed_samples)
            
            #Make prediction with sampled weights.
            predictions = self.sess.run(self.model, feed)
                
            #Average all predictions
            prediction= np.mean(predictions,axis=0)
            #prediction= predictions[0]
            
            #Retireve mu and std
            mu= self.sess.run(self.v_params['mu'].values())
            std=self.sess.run(self.v_params['std'].values())            
        else:
            prediction= self.sess.run(self.model, feed)
                
        return prediction
    
X_t= X[:V]
Y_t= Y[:V]

Z_w1_t= Z_w1[:V]
Z_b1_t= Z_b1[:V]
Z_w2_t= Z_w2[:V]
Z_b2_t= Z_b2[:V]

nn = nnet(M, H, O, bayes= True)
nn.fit(X_t, Y_t, Z_w1_t, Z_b1_t, Z_w2_t, Z_b2_t, epochs= 5000)

X_v= X[V:]
Y_v= nn.predict(X_v, n= 100)

display(X_t, Y_t, X_v, Y_v)