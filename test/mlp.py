# modules
# ML
import sys
import autograd.nn as nn
import autograd.optim as optim
import autograd.functions as F
import autograd.tensor as tensor
# miscellaneous
from typing import Callable


# function definitions
def train(nn : nn.MLP, loss_fn : Callable, optimizer : optim.Optimizer, xs : tensor.Tensor, ys : list, epochs= 300, lr= 0.1, flag= True):
    
    """
    flag should be set when the optimizer is SGD & unset when it is ADAM.
    """
    print('Target:', ys)
    # machine learning
    if flag:
        optimizer= optimizer(nn, lr)
    else:
        optimizer= optimizer(nn, lr, [0.9, 0.999])
        
    for i in range(epochs+1):
        # generate the predcitions
        ps= [nn(xs[i, :]) for i in range(len(xs))]
        # compute the loss
        loss= loss_fn(ps, ys)
        # pretty print
        if i % 25 == 0:
            print(f'\nEpoch {i}, loss: {round(loss.val, 3)}')
            print('predictions:',list(map(lambda x : round(x.val, 3), ps)))
        # [back-]propagate the gradients
        loss.backward()
        # update weights & biases
        optimizer.step()
        # reset gradients
        optimizer.zero_grad()

# main
# 1. create the datset
xs= tensor.Tensor(
    [[0, 0, 0, 1],
     [0, 0, 1, 0],
     [0, 0, 1, 1],
     [0, 1, 0, 0]]
    )

ys=  [0, 0, 1, 1]


# tests
# to specify the input-size of a neural-network
input_size= xs.shape[1]

# 1. classification problem
classifier= nn.MLP(input_size, (input_size>>1, input_size>>2), ('relu', 'sigmoid',))
print('1. Task: Classification. Optimizer: ADAM.', classifier, end='\n\n')
train(classifier, F.BCE, optim.ADAM, xs, ys, 150, 0.1, False)
print('\nComment: In case the classification-model did not converge, re-run this script. The weight-initialization affects convergence!')

# 2. regression problem
regressor= nn.MLP(input_size, (1,), ('none',))
print('\n2. Task: Regression. Optimizer: ADAM.', regressor, end='\n\n')
train(regressor, F.MSE, optim.ADAM, xs, ys, 100, 0.1, False)
