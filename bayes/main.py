import data
from data.visualization import Visualizer
from bnn import NNet

import argparse

"""
Trains deterministic and bayesian neural networks for a simple regression task
on synthesized dataset.
"""

parser = argparse.ArgumentParser()
# Data arguments
parser.add_argument('--dataset',type=str,default='pringle')
parser.add_argument('--data_std',type=float,default=0.2) # determines noisiness of dataset.

# Training arguments
parser.add_argument('--epochs',type=int,default=100)
parser.add_argument('--n_examples',type=int,default=1000)
parser.add_argument('--percent_train',type=float,default=0.7)

# Network arguments
parser.add_argument('--use_bayes',type=bool,default=False) # use bayesian neural network
parser.add_argument('--n_input',type=int,default=2) # feature dim / input layer size
parser.add_argument('--n_hidden',type=int,default=5)
parser.add_argument('--n_output',type=int,default=1) # note: visualizer only displays 1D predictions

# Hyperparameter arguments
parser.add_argument('--batch_size',type=int,default=10)
parser.add_argument('--learning_rate',type=float,default= 1e-2)
parser.add_argument('--reg',type=float,default=0.0) # l2 (frequentist) regularization to prevent overfitting

# Visualization arguments
parser.add_argument('--interval',type=int,default=15)
parser.add_argument('--grow_epochs',type=bool,default=False) # dont set epoch axis limit.

args = parser.parse_args()

try:
    dataset = data.regression_data[args.dataset]
except KeyError:
    raise KeyError("Dataset '{}' does not exist.".format(args.dataset))

X, Y = data.pringle(args.n_examples,args.n_input,args.n_output,std=args.data_std)
V = int(args.percent_train * args.n_examples)

X_t= X[:V]
Y_t= Y[:V]
X_v= X[V:]
Y_v= Y[V:]

if args.grow_epochs:
    epochs= None
else:
    epochs= args.epochs
    
vis = Visualizer(dim= args.n_input,
                 xlim=(X.min(),X.max()),ylim=(Y.min(),Y.max()),
                 epochs= epochs, interval= args.interval)
nn = NNet(vis, args.n_input, args.n_hidden, args.n_output, bayes=args.use_bayes,
          batch_size= args.batch_size, learning_rate= args.learning_rate)
nn.fit(X_t, Y_t, X_v, Y_v, epochs= args.epochs)
