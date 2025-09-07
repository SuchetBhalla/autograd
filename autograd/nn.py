import random
from .engine import Value

class Module:
    
    def _parameters(self):
        """desc.: for internal use: this function must return the paramaters of the entire Network/Layer/Neuron as one list,
        which shall be parsed to reset the gradient with the function 'zero_grad'"""
        return []
    
    def zero_grad(self):
        """desc.: reset gradient of the Network/Layer/Neuron"""
        for p in self._parameters():
            p.grad = 0

class Neuron(Module):
    
    def __init__(self, input_size, nonlinearity):
        assert nonlinearity in ('none', 'relu', 'tanh', 'sigmoid'), "Non-linearity must be one of ('none', 'relu', 'tanh', 'sigmoid')"
        self.nonlinearity= nonlinearity
        self.b= Value(random.gauss(0, 1/input_size), (), '', True)
        self.w= [Value(random.gauss(0, 1/input_size), (), '', True) for _ in range(input_size)]
                
    def __repr__(self):
        return f'Neuron:: No.of parameters: {len(self.w) + 1}, non-linearity: {self.nonlinearity}'
    
    def _parameters(self):
        return self.w + [self.b]
    
    def __call__(self, x):
        assert len(x) == len(self.w), f'Expected an input-array of size {len(self.w)}'
        
        # compute the affine sum: x*w + b
        ip= [w*x for w, x in zip(self.w, x)]
        s= sum(ip, self.b)
        
        # activation function
        if self.nonlinearity == 'relu':
            out= s.relu()
        elif self.nonlinearity == 'sigmoid':
            out= s.sigmoid()
        elif self.nonlinearity == 'tanh':
            out= s.tanh()
        else:
            out= s
            
        return out
    
class Layer(Module):
    def __init__(self, input_size, output_size, nonlinearity):
        assert nonlinearity in ('none', 'relu', 'tanh', 'sigmoid'), "Non-linearity must be one of ('none', 'relu', 'tanh', 'sigmoid')"
        self.nonlinearity= nonlinearity
        self.neurons= [Neuron(input_size, nonlinearity) for _ in range(output_size)]
    
    def __repr__(self):
        output_size= len(self.neurons)
        parameters = len(self.neurons[0].w) + 1 # a 'Neuron' is an object
        return f'Layer:: No. of parameters: {output_size*parameters}, non-linearity: {self.nonlinearity}'
    
    def _parameters(self):
        """desc.: for internal use: to reset the gradients [of the parameters] of each 'Neuron'"""
        return [p for n in self.neurons for p in n._parameters()]
    
    def parameters(self):
        """desc.: to view the parameters of each neuron; creates a list[list[Values]]"""
        return [n._parameters() for n in self.neurons]
        
    def __call__(self, x):
        out= [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
class MLP(Module):
    def __init__(self, sizes, nonlinearities):
        """
        desc.: creates an MLP
        arguments:
            sizes : tuple : (input_size, output_size_1,   output_size_2,   ... output_size_n)
            nonlinearities : tuple :  (nonlinearlity_1, nonlinearlity_2, ... nonlinearlity_n)
        """
        assert len(sizes) == len(nonlinearities) + 1, f'Specify a non-linearity per layer; For the current inputs, the number of layers & nonlinearities are ({len(sizes)-1}, {len(nonlinearities)})'
        self.nn= [Layer(sizes[i], sizes[i+1], nonlinearities[i]) for i in range(len(sizes)-1)]
        
    def __repr__(self):
        # count the total-number of learnable-parameters in the network
        total= 0
        parameters= len(self.nn[0].neurons[0].w) + 1 # layer 1's input_size; +1 for bias
        for i in range(len(self.nn)):
            output_size= len(self.nn[i].neurons)
            total += output_size * parameters
            parameters= output_size + 1
        # collect names of the nonlinearities
        nonlinearities= [layer.nonlinearity for layer in self.nn]
        return f'MLP: No. of parameters: {total}, non-linearities: {"->".join(nonlinearities)}'
    
    def _parameters(self):
        """desc.: for internal use: to reset the gradients [of the parameters] of each 'Neuron' [in each 'Layer']"""
        return [p for l in self.nn for p in l._parameters()]
    
    def parameters(self):
        """desc.: to view the parameters of each layer; creates list[list[list[Values]]]"""
        return [l.parameters() for l in self.nn]
    
    def print_network(self): # pretty printing
        """desc.: pretty printing to view the parameters of each layer"""
        print('-'*10)
        print('Network: [\n')
        for i, ly in enumerate(self.parameters()):
            print(f'Layer {i+1}: dim(input): {len(ly[0])-1}, dim(output): {len(ly)}\n[') # -1, because dim(input) = dim(weights) - dim(bias)
            if len(ly) > 1:
                for n in ly:
                    print(n)
            else:
                print(ly)
            print(f']; end of layer\nnon-linearity used: {self.nn[i].nonlinearity}\n')
        print('\n]; end of Network')
        print('-'*10)
    
    def __call__(self, x):
        """desc.: forward pass"""
        for layer in self.nn:
            x= layer(x) # the scope of 'x' is local to this function; thus the pointer [to the input] 'x' is not modified
        return x