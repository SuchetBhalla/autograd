from .engine import Value

def _util_loss(predictions, targets):
    """
    desc.: this function verifies the data-type of the objects in the lists 'predictions' & 'targets'.
    arguments:
        predictions : list[Value]
        targets     : list
    """
    # error check
    assert len(predictions) == len(targets), "Length of predictions must equal targets"
    if not all(isinstance(p, Value) for p in predictions):
        assert False, "All elements of the list 'predictions' must be of type 'Value'"

    targets= [Value(t) if not isinstance(t, Value) else t for t in targets]
    n = len(targets)
    pt= zip(predictions, targets)
    return n, pt

def MSE(predictions, targets):
    n, pt = _util_loss(predictions, targets)
    return sum([(p-t)**2 for p, t in pt])/n

def BCE(predictions, targets):
    """refer: https://en.wikipedia.org/wiki/Cross-entropy"""
    n, pt = _util_loss(predictions, targets)
    return -sum([p.log_2() if t.val else (1-p).log_2() for p,t in pt])