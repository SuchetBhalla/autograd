import random
from math import floor, sqrt
from .engine import Value
from .tensor import Tensor

class Module:

    def __init__(self):
        """desc.: base class for all neural networks"""
        pass

    def _parameters(self):
        """desc.: for internal use: this function must return the paramaters of the neural network as one list,
        which shall be parsed, to reset the gradients with the function 'zero_grad'"""
        
        # returns pointers to all neural-networks from 'self'
        submodules= self.get_submodules()
        
        # returns (pointers to) the parameters as 1 list
        params= []
        for nn in submodules:
            params.extend(nn._parameters())
        
        return params

    def zero_grad(self):
        """desc.: reset gradients [of the parameters] of the neural-network"""
        for p in self._parameters():
            p.grad = 0
            
    def __len__(self):
        """desc.: returns the number of layers in the neural-network."""
        submodules= self.get_submodules()
        count= 0
        for nn in submodules:
            count += len(nn)
        return count
    
    def __repr__(self):
        return f'No. of layers {len(self)}'
    
    def get_submodules(self):
        """
        desc.: reads the attributes of 'self' & extracts the objects of type 'Module'.
        The objective is to extract the neural-networks from 'self'
        """
        submodules= []
        for attr in dir(self):
            if not attr.startswith('__') and isinstance(getattr(self, attr), Module):
                submodules.append(getattr(self, attr))
        return submodules

class Neuron(Module):

    def __init__(self, input_size, nonlinearity):
        """
        desc.:
        creates:
            w : list[Value]
            b : Value
        these parameters are initialized randomly, with values sampled from a Gaussian-distribution, with (mean, std) = (0, 1/input_size)

        input:
            input_size : int
            nonlinearity : str

        info:
            A "neuron" is composed of: weights 'w', 1 bias 'b' & 1 non-linear function 'f'.
            It computes,
                1. an affine sum: z = x*w + b
                2. a non-linear transformation: y = f(z)
            The output is 'y'.

        initialization of parameters: https://www.youtube.com/watch?v=s2coXdufOzEs
        """
        assert input_size > 0, "Input size must be a positive integer."
        assert nonlinearity in ('none', 'relu', 'tanh', 'sigmoid'), "Non-linearity must be one of ('none', 'relu', 'tanh', 'sigmoid')"
        self.nonlinearity= nonlinearity
        self.b= Value(random.gauss(0, 2/input_size), (), '', True)
        self.w= [Value(random.gauss(0, 2/input_size), (), '', True) for _ in range(input_size)]

    def __len__(self):
        return len(self.w) +1 # +1 for the bias

    def __repr__(self):
        return f'Neuron:: No.of parameters: {len(self)}, non-linearity: {self.nonlinearity}' # +1 for the bias

    def _parameters(self):
        """
        desc.: returns a list of pointers, to the learnable-parameters [of the Neuron]

        returns list[Value]
        """
        return self.w + [self.b]

    def __call__(self, x):
        """
        desc.: forward pass
        input: x : list[Value]
        returns Value
        """
        assert len(x) == len(self.w), f'Expected an input-array of size {len(self.w)}, got {len(x)}'
        
        # compute the affine sum: x*w + b
        ip= [w*e for w, e in zip(self.w, x)]
        s= sum(ip, self.b)

        # activation function: performs a non-linear operation on the affine-sum
        func= getattr(s, self.nonlinearity, None)
        return func() if callable(func) else s # func() -> Value

class Layer(Module):
    def __init__(self, input_size, output_size, nonlinearity):
        """
        desc.: a 'Layer' is composed of a stack of 'Neuron'.
        creates:
            neurons : list[Neuron]
        input:
            input_size: int ; size of input vector
            output_size: int; size of output vector
            nonlinearity: str
        """
        assert all(isinstance(x, int) and x > 0 for x in [input_size, output_size]), 'the vector-sizes must be positive integers'
        self.neurons= [Neuron(input_size, nonlinearity) for _ in range(output_size)]
        self.nonlinearity= nonlinearity

    def __len__(self):
        return len(self.neurons)

    def __repr__(self):
        output_size= len(self)
        parameters = len(self.neurons[0].w) +1 # a 'Neuron' is an object; +1 for the bias
        return f'Layer:: No. of parameters: {output_size*parameters}, non-linearity: {self.nonlinearity}'

    def _parameters(self):
        """desc.: returns a list of pointers, to the parameters of the neurons, in this layer
        utility: to reset the gradients of the parameters
        returns list[Value]"""
        return [p for n in self.neurons for p in n._parameters()]

    def parameters(self): # pretty printing
        """desc.: to view the parameters of each neuron
        returns list[list[Values]]"""
        return [n._parameters() for n in self.neurons]

    def __call__(self, x):
        """
        desc.: forward pass
        input: x : list[Value]
        returns Union(Value, list[Value])
        """
        out= [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

class MLP(Module):
    def __init__(self, input_size, output_sizes, nonlinearities):
        """
        desc.: creates a Multi-Layer Perceptron, i.e., a stack of 'Layer'
        creates:
            nn : list[Layer]
        input:
            sizes : tuple : (input_size, output_size_1,   output_size_2,   ... output_size_n)
            nonlinearities : tuple :  (nonlinearlity_1, nonlinearlity_2, ... nonlinearlity_n)
        """
        assert len(output_sizes) == len(nonlinearities), f'Specify a non-linearity per layer; For the current inputs, the number of layers & nonlinearities are ({len(output_sizes)}, {len(nonlinearities)})'
        sizes= [input_size] + list(output_sizes)
        self.nn= [Layer(sizes[i], sizes[i+1], nonlinearities[i]) for i in range(len(output_sizes))]

    def __len__(self):
        return len(self.nn)

    def parameter_count(self):
        """desc.: returns a count of the total-number of learnable-parameters in the network"""
        total= 0
        parameters= len(self.nn[0].neurons[0].w) +1 # layer 1's input_size; +1 for bias
        for i in range(len(self)):
            output_size= len(self.nn[i].neurons)
            total += output_size * parameters
            parameters= output_size + 1
        return total

    def __repr__(self):
        """prints: the total number of parameters in the MLP & the sequence of nonlinearities"""
        # collect names of the nonlinearities
        nonlinearities= [layer.nonlinearity for layer in self.nn]
        return f'MLP: No. of parameters: {self.parameter_count()}, non-linearities: {"->".join(nonlinearities)}'

    def _parameters(self):
        """desc.: returns a list of pointers, to the parameters of the neurons, in this MLP
        utility: to reset the gradients of the parameters
        returns list[Value]"""
        return [p for l in self.nn for p in l._parameters()]

    def parameters(self): # pretty printing
        """desc.: to view the parameters of each layer
        returns list[list[list[Values]]]"""
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
        """desc.: forward pass
        input: x : Tensor
        returns: Union(Value, list[Value])
        """
        assert isinstance(x, Tensor) and len(x.shape) == 1, 'Input must be a 1D array of type \'Tensor\''
        x= x.tensor # extracts list[Value] from Tensor
        for layer in self.nn:
            x= layer(x) # the scope of 'x' is local to this function; thus the pointer [to the input] 'x' is not modified
        return x


class Kernel(Neuron):
    def __init__(self, input_size, nonlinearity):
        """desc.: This class is equivalent to a 'Neuron'"""
        super().__init__(input_size, nonlinearity)

class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, nonlinearity, stride= 1, padding= 0, bias= True):
        """
        desc.: WIP: this class is akin to a 'Layer'
        inputs:
            in_channels : int
            out_channels: int
            kernel_size: int
            nonlinearity: str
            stride: int
            padding: int
            bias: boolean
        creates:
            neurons : list[Neurons] //Neuron == Kernel
        '"""
        assert all(isinstance(x, int) and x > 0 for x in [in_channels, out_channels, kernel_size]), 'the channels & kernel-size must be positive integers'
        assert nonlinearity in ('none', 'relu', 'tanh', 'sigmoid'), "Non-linearity must be one of ('none', 'relu', 'tanh', 'sigmoid')"
        assert stride == 1 and padding == 0 and bias== True, "currently, support is limited to stride == 1 and padding == 0 and bias == True"

        self.bias = bias
        self.padding= padding
        self.stride = stride
        self.nonlinearity= nonlinearity
        self.kernel_size= kernel_size
        self.out_channels= out_channels
        self.in_channels= in_channels

        # create kernels i.e., neurons
        # refer: https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html
        assert kernel_size == 3, "currently, support is limited a symmetric-kernel of size 3."
        self.neurons= [Kernel(kernel_size**2 * in_channels, nonlinearity) for _ in range(out_channels)]

    def __call__(self, x):
        """
        desc.: forward pass
        input : x : Tensor
        returns Tensor
        """

        # 0. init
        (channels, rows, cols) = x.shape
        assert channels == self.in_channels, f"Expected {self.in_channels} channels as input. Received {channels} channels."
        ks, s= self.kernel_size, self.stride

        # 1. perform 2D convolution on the image
        ## this list shall store the "activation maps", as [1d] vectors. eventually each vector shall be converted to a [2d] matrix.
        activation_map= [[] for _ in range(self.out_channels)] # fact: self.out_channels == len(self.neurons) i.e., no. of kernels
        r, c= rows-ks+1, cols-ks+1
        for i in range(0, r, s):
            for j in range(0, c, s):
                # 1. read a 3D sub-tensor; akin to a sub-matrix
                sub= x[:, i:i+ks, j:j+ks]
                sub.flatten() # 3d -> 1d; because Kernel == Neuron
                vec= sub.tensor # Tensor -> list[Value]
                # 2. forward-pass
                for k, kernel in enumerate(self.neurons):
                    activation_map[k].append(kernel(vec))

        # 2. reshape the activation_map; 2D -> 3D
        # a. compute expected dimensions of the activation map
        # refer: https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html
        out_rows= floor((rows + self.padding*2 - ks + s)/s)
        out_cols= floor((cols + self.padding*2 - ks + s)/s)
        assert out_rows*out_cols == len(activation_map[0]), f"len(activation_map) != out_rows*out_cols"

        # b. reshape: 2D -> 3D
        activation_map= Tensor(activation_map)
        activation_map.reshape((self.out_channels, out_rows, out_cols))

        return activation_map

class CNN(MLP):
    def __init__(self, in_channels, out_channels, kernel_sizes, nonlinearities):
        """
        desc.: this class is akin to 'MLP'
        inputs:
            in_channels : int
            out_channels : list[int]
            kernel_sizes : list[int]
            nonlinearities : list[str]
        """
        assert len(out_channels) == len(nonlinearities) and len(out_channels) == len(kernel_sizes), f'len(out_channels) != len(nonlinearities) and len(out_channels) != len(kernel_sizes)'
        channels= [in_channels] + list(out_channels)
        self.nn= [Conv2d(channels[i], channels[i+1], kernel_sizes[i], nonlinearities[i]) for i in range(len(out_channels))]

    def __repr__(self):
        """prints: the total number of parameters in the MLP & the sequence of nonlinearities"""
        # collect names of the nonlinearities
        nonlinearities= [layer.nonlinearity for layer in self.nn]
        return f'CNN: No. of parameters: {self.parameter_count()}, non-linearities: {"->".join(nonlinearities)}'

    def __call__(self, x):
        """desc.: forward pass
        input: x : Tensor
        returns: Tensor
        """
        assert isinstance(x, Tensor), 'Input must of type \'Tensor\''
        for layer in self.nn:
            x= layer(x) # the scope of 'x' is local to this function; thus the pointer [to the input] 'x' is not modified
        return x
