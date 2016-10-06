import data
from data.visualization import Visualizer
from bnn import NNet

import argparse

parser = argparse.ArgumentParser()

# Data arguments
parser.add_argument('--data_std',type=float,default=0.2)

# Training arguments
parser.add_argument('--epochs',type=int,default=100)
parser.add_argument('--n_examples',type=int,default=1000)
parser.add_argument('--percent_train',type=float,default=0.7)

# Network arguments
parser.add_argument('--use_bayes',type=bool,default=False) # use bayesian neural network
parser.add_argument('--n_input',type=int,default=2) # feature dim / input layer size
parser.add_argument('--n_output',type=int,default=1)
parser.add_argument('--n_hidden',type=int,default=5)

# Visualization arguments

args = parser.parse_args()

X, Y = data.pringle(args.n_examples,args.n_input,std=args.data_std)
V = int(args.percent_train * args.n_examples)

X_t= X[:V]
Y_t= Y[:V]
X_v= X[V:]
Y_v= Y[V:]

vis = Visualizer(dim= 2, xlim=(X.min(),X.max()),ylim=(Y.min(),Y.max()), epochs= None, interval= 15)
nn = NNet(vis, args.n_input, args.n_hidden, args.n_output, bayes= args.use_bayes)
nn.fit(X_t, Y_t, X_v, Y_v, epochs= args.epochs)

Y_p= nn.predict(X_v, n= 100)