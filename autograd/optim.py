from .nn import Module
from math import sqrt

class Optimizer:
    def __init__(self, nn, lr):
        """
        desc.: This class shall be the parent-class for all gradient-descent-algorithms.
        arguments:
            nn : Module
            lr : float ; the step-size
        """
        assert isinstance(nn, Module), "A Neural Network must be an instance of the class 'Module'"
        assert callable(getattr(nn, '_parameters', None)), 'The class-definition of a neural-network must contain the method \'_parameters\''
        assert isinstance(lr, float) and lr > 0, 'The learning rate must be a positive float'
        self.lr=lr
        self.parameters= nn._parameters()

    def zero_grad(self):
        """desc.: reset the gradients"""
        for p in self.parameters:
            p.grad= 0

class SGD(Optimizer):
    def __init__(self, nn, lr):
        """desc.:
            algorithm: Stochastic Gradient Descent"""
        super().__init__(nn, lr)

    def step(self):
        """desc.: adjust the weights & biases"""
        for p in self.parameters:
            p.val -= self.lr*p.grad

class ADAM(Optimizer):
    def __init__(self, nn, lr, betas, epsilon= 10**-8):
        """
        desc.:
            ADAM:
            expansion: ADAptive Moment estimation
            utility: an algorithm for [first-order-] gradient-based optimization of an objective-function.
            details: Estimates of the first & second [raw-]moments of the gradient, are used to modify [aka adapt] the stepsize; per parameter [per iteration].
        input:
            nn : Module
            lr : float ; aka step-size
            betas : list[float] = [beta_1, beta_2] ; decay rates
            epsilon : float ; a constant to prevent division by zero
        references:
            1. https://arxiv.org/abs/1412.6980
            2. https://en.wikipedia.org/wiki/Moment_(mathematics)
        """
        super().__init__(nn, lr)

        # error checks
        assert isinstance(epsilon, float) and epsilon > 0, "Epsilon must be a positive float"
        if not all(isinstance(beta, float) and 0 < beta < 1 for beta in betas):
            assert False, "Both betas must lie in the range (0, 1)"
        

        # \beta, (1-\beta)
        self.b1, self.b1_c = betas[0], 1-betas[0]
        self.b2, self.b2_c = betas[1], 1-betas[1]
        
        self.epsilon= epsilon
        
        # init for ADAM
        self.n= len(self.parameters)
        self.m_t= [0]*self.n
        self.v_t= [0]*self.n
        self.t= 0

    def step(self):
        """desc.: adjust the weights & biases"""
        self.t +=1
        for i in range(self.n):
            g= self.parameters[i].grad
            g2= g**2
            # compute the [biased,] exponentially-decaying moving-average of the first & second raw-moments of the gradient
            self.m_t[i]= self.b1*self.m_t[i] + self.b1_c*g
            self.v_t[i]= self.b2*self.v_t[i] + self.b2_c*g2
            # bias correction
            alpha_t = self.lr * sqrt(1-self.b2**self.t) / (1-self.b1**self.t)
            # adjust the parameter
            self.parameters[i].val -= alpha_t*self.m_t[i]/(sqrt(self.v_t[i])+self.epsilon)
