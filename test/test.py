# modules
# ML
from autograd.nn import MLP
from autograd.optim import BGD
from autograd.loss_functions import BCE, MSE
# miscellaneous
from typing import Callable


# function definitions
def train(nn : MLP, loss_fn : Callable, xs : list, ys : list, epochs= 300, lr= 0.1):
    print('Target:', ys)
    # machine learning
    optimizer= BGD(nn, lr)
    for i in range(epochs+1):
        # generate the predcitions
        ps= [nn(x) for x in xs]
        # compute the loss
        loss= loss_fn(ps, ys)
        # pretty print
        if i % 50 == 0:
            print(f'\nEpoch {i}, loss: {round(loss.val, 3)}')
            print('predictions:',list(map(lambda x : round(x.val, 3), ps)))
        # [back-]propagate the gradients
        loss.backward()
        # update weights & biases
        optimizer.step()
        # reset gradients
        optimizer.zero_grad()

# main
# define the datset
xs= [[0, 0, 0, 1],
     [0, 0, 1, 0],
     [0, 0, 1, 1],
     [0, 1, 0, 0]]
ys=  [0, 0, 1, 1]

# to specify the input-size of a neural-network
input_size= len(xs[0])

# tests
# 1. classification problem
# define the network
classifier= MLP((input_size, input_size>>1, input_size>>2),
                ('relu', 'sigmoid',))
print('1. Classifier:', classifier, end='\n\n')
train(classifier, BCE, xs, ys, 300, 0.1)

# 2. regression problem
regressor= MLP((input_size, 1,), ('none',))
print('\n2. Regressor:', regressor, end='\n\n')
train(regressor, MSE, xs, ys, 300, 0.1)