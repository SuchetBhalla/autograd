from .nn import MLP

class BGD:
    def __init__(self, nn, lr):
        """
        desc.: Batch Gradient Descent
        arguments:
            nn : MLP
            lr : float
        """
        assert isinstance(nn, MLP), "A Neural Network must be an instance of the class 'MLP'"
        assert isinstance(lr, float), 'The learning rate must be a float'
        self.lr =lr
        self.parameters= nn._parameters()
        
    def zero_grad(self):
        """desc.: reset the gradients"""
        for p in self.parameters:
            p._grad= 0
    
    def step(self):
        """desc.: adjust the weights & biases"""
        for p in self.parameters:
            p.val -= self.lr*p._grad