from math import exp, log2

class Value:
    def __init__(self, value, prev= (), op= '', requires_grad= False):
        assert isinstance(value, (int, float)), "Data-type must be int/float"
        self.val= value
        self._prev= set(prev)
        self._op= op
        self._grad= 0
        self._backward= lambda: None
        self.requires_grad= requires_grad
        
    def __repr__(self):
        return f'Value: {self.val}, Grad: {self._grad}'
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        
        out= Value(self.val + other.val, (self, other,), '+', self.requires_grad or other.requires_grad)
        def _back_propagate():
            """
            desc.: backpropagate the gradient to the previous nodes.
            
            the gradients are computed by using the chain rule.
            Let, A = f(B) & B= g(C)
            chain rule: dA/dC = dA/dB * dB/dC
            
            For addition:
            Let,
                A = B + C
                Loss = f(A)
                
            dLoss/dB = dLoss/dA * dA/dB
                     = dLoss/dA * 1
            """
            if self.requires_grad:
                self._grad += out._grad
            if other.requires_grad:
                other._grad+= out._grad
        
        #This function-pointer is a 'closure', consequently the variables 'self' & 'other' are stored with it
        out._backward= _back_propagate
        return out
    
    def __radd__(self, other): # other + self -> self + other
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.val * other.val, (self, other,), '*', self.requires_grad or other.requires_grad)
        def _back_propagate():
            if self.requires_grad:
                self._grad += out._grad * other.val
            if other.requires_grad:
                other._grad+= out._grad * self.val
        out._backward= _back_propagate
        return out
    
    def __rmul__(self, other): # other * self -> self * other
        return self * other
    
    def __neg__(self): # -self -> self * -1
        return self * -1
    
    def __sub__(self, other): # self - other -> self + (-other)
        return self + -other
    
    def __rsub__(self, other): # other - self -> -self + other
        return -self + other
    
    def __pow__(self, other): # self ** other
        assert isinstance(other, (int, float)), "Data-type must be int/float"
        out= Value(self.val**other, (self,), f'^{other}', self.requires_grad)
        def _back_propagate():
            if self.requires_grad:
                self._grad+= out._grad * other * self.val**(other-1)
        out._backward= _back_propagate
        return out
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other): # other / self -> other * self**-1
        return other * self**-1
    
    
    def _exp(self, multiplier= 1):
        x= self.val*multiplier
        return exp(x) if x < 4.6 else 100

    def log_2(self):
        val= log2(self.val) if self.val > 1e-10 else -33.2
        out= Value(val, (self,), 'log2', self.requires_grad)
        def _back_propagate():
            if self.requires_grad:
                val = 1/self.val if self.val > 1e-10 else 1e10
                self._grad += out._grad*val
        out._backward= _back_propagate
        return out

    def sigmoid(self):
        out = Value(1 / (1 + self._exp(-1)), (self,), 'sigmoid', self.requires_grad)
        def _back_propagate():
            if self.requires_grad:
                self._grad += out._grad * out.val * (1-out.val)
        out._backward= _back_propagate
        return out
    
    def tanh(self):
        e2x = self._exp(2)
        val= (e2x - 1) / (e2x + 1)
        out = Value(val, (self,), 'tanh', self.requires_grad)
        def _back_propagate():
            if self.requires_grad:
                self._grad += out._grad * (1- out.val**2)
        out._backward= _back_propagate
        return out
    
    def relu(self):
        out= Value(self.val if self.val > 0 else 0, (self,), 'relu', self.requires_grad)
        def _back_propagate():
            if self.requires_grad:
                self._grad += out._grad if out.val > 0 else 0
        out._backward= _back_propagate
        return out
    
    def topological_sort(self, visited, _list):
        if self not in visited:
            visited.add(self)
            for v in self._prev:
                v.topological_sort(visited, _list)
            _list.append(self)
    
    def backward(self):
        # perform topological sort
        visited, _sorted= set(), list()
        self.topological_sort(visited, _sorted)
        _sorted.reverse()
        
        # the derivative of the root w..r.t the root is 1
        self._grad= 1
        for v in _sorted:
            if v.requires_grad:
                v._backward()