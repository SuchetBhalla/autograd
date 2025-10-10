from .engine import Value
from typing import Union
from math import e, log

def _util_func(predictions : list[Value], targets : list[Union[int, float, Value]]) -> (int, zip):
    """
    desc.: this function verifies the data-type of the objects in the lists 'predictions' & 'targets'.
    """
    # error check
    assert len(predictions) == len(targets), "length of predictions must equal targets"
    assert all(isinstance(p, Value) for p in predictions), "All elements of the list 'predictions' must be of type 'Value'"

    targets= [Value(t) if not isinstance(t, Value) else t for t in targets]
    
    n = len(targets)
    pt= zip(predictions, targets)
    return n, pt

# loss functions
def MSE(predictions, targets) -> Value:
    n, pt = _util_func(predictions, targets)
    return sum([(p-t)**2 for p,t in pt])/n

def BCE(predictions, targets) -> Value:
    """refer: https://en.wikipedia.org/wiki/Cross-entropy"""
    n, pt = _util_func(predictions, targets)
    return -sum([p.ln() if t.val else (1-p).ln() for p,t in pt])/n

def log_softmax(logits : list[Value], target : int) -> Value:
    """
    desc.: aka log-of-likelihood. e.g.,
    1. softmax(a) = e**a / (e**a + e**b + e**c)
    2. log(softmax(a))
    3. f(a,b,c) = log(softmax(a))
                = a - log(e**a + e**b + e**c)
    4. gradients:
    df/da = 1 - 1/(e**a + e**b + e**c)*(e**a) = 1-softmax(a)
    df/db = 0 - 1/(e**a + e**b + e**c)*(e**b) =  -softmax(b)
    df/dc = -softmax(c)
    """
    # extract values
    values= list(map(lambda v: v.val, logits))    
    
    # shift values; for numerical stability[?]
    m= max(values)
    values= list(map(lambda v: v-m, values))
    
    # compute log_softmax [for each element in the list 'values']
    den= log(sum(list(map(lambda v: e**(v), values))))
    log_softmax= [v-den for v in values]
    
    # output; choose the target
    out= Value(log_softmax[target], tuple(logits), 'log_softmax', all(x.requires_grad for x in logits))
    
    def _back_propagate():
        softmax= [e**x for x in log_softmax]
        for i, x in enumerate(logits):
            if x.requires_grad:
                delta= 1 if i == target else 0
                x.grad += out.grad * (delta-softmax[i])
    out._backward= _back_propagate
    return out

def NLL(preds : list[list[Value]], targets : list[int]) -> Value:
    """
    desc.: computes negative loss of likelihood i.e.,
    -1/N * (\sum^N_{i=0} log_softmax(preds[i], targets[i]))
    where N = len(preds)-1
    """
    # error checks
    assert len(preds) == len(targets), 'len(preds) != len(targets)'
    assert all(len(logits) == len(preds[0]) for logits in preds[1:]), "each list in 'preds' must be of the same length"
    assert all(isinstance(p, Value) for logits in preds for p in logits), "each value in 'preds' must be of type 'Value'"
    assert all(isinstance(t, int) and -1 < t < len(preds[0]) for t in targets), f"each value in 'targets' must be an integer, in the range [0, {len(preds[0])})"
    
    return -sum([log_softmax(logits, target) for logits, target in zip(preds, targets)]) / len(targets)


# utilities
def argmax(arr : list[Value]) -> int:
    """desc.: returns the index of the max. element in the list 'arr'"""
    # list[Value] -> list[float]
    arr= [x.val if isinstance(x, Value) else x for x in arr]
    # argmax
    arg_max, Max= 0, arr[0]
    for i, e in enumerate(arr):
        if e > Max:
            arg_max, Max = i, e
    return arg_max
    