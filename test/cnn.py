# modules
import autograd.functions as F
import autograd.tensor as tensor
import autograd.optim as optim
import autograd.nn as nn


# main
# 1. Here I create the dataset
# xs, ys = [x, +, o], [0, 1, 2]
ys= [0, 1, 2]

xs=[
    [[1, 0, 0, 0, 1],
     [0, 1, 0, 1, 0],
     [0, 0, 1, 0, 0],
     [0, 1, 0, 1, 0],
     [1, 0, 0, 0, 1]],

    [[0, 0, 1, 0, 0],
     [0, 0, 1, 0, 0],
     [1, 1, 1, 1, 1],
     [0, 0, 1, 0, 0],
     [0, 0, 1, 0, 0]],

    [[1, 1, 1, 1, 1],
     [1, 0, 0, 0, 1],
     [1, 0, 0, 0, 1],
     [1, 0, 0, 0, 1],
     [1, 1, 1, 1, 1]],
   ]


# convert 'xs' list[list[int]] to Tensor
xs= list(map(lambda x: tensor.Tensor(x), xs))
# convert 2D matrices to 3D tensors; because a CNN (currently) accepts only 3D tensors
[x.reshape((1, *x.shape)) for x in xs if len(x.shape) == 2]


# 2. define the network
class NeuralNet(nn.Module):
    
    def __init__(self):
        """desc.: the predictions shall be logits"""
        self.cnn= nn.CNN(in_channels= xs[0].shape[0], out_channels= (8, 8,), kernel_sizes= (3, 3), nonlinearities= ('relu', 'relu'))
        self.nn = nn.MLP(input_size= 8, output_sizes= (4, len(ys), ), nonlinearities= ('relu', 'none'))
        
    def __call__(self, x):
        x= self.cnn(x)
        x.flatten() # flattens inplace
        x= self.nn(x)
        return x
    
# 3. an experiment to test convergence: can the model overfit the dataset?
model= NeuralNet()
print(model, '.', 'Model\'s architecture:\n', model.cnn, '\n', model.nn, '\n')

# train
optimizer= optim.ADAM(model, 0.1, [0.9, 0.999])
for i in range(10):
    # generate predictions; outputs logits
    ps= [model(x) for x in xs]
    # compute [scalar-]loss
    loss= F.NLL(ps, ys)
    # backpropagate the gradients
    loss.backward()
    # adjust the parameters
    optimizer.step()
    # reset gradients
    optimizer.zero_grad()
    
    # pretty printing
    print(f'\nEpoch {i}, loss: {round(loss.val, 3)}')
    print('target:', ys)
    print('predictions:', [F.argmax(p) for p in ps])
    
print('\nComment: In case the model did not converge, re-run this script. The weight-initialization affects convergence!')
