"""
Desc.: This script defines the class 'Tensor', which accepts [as input] an n-dimensional list, and associates [with it] methods to reshape that list. 
This class has utility in [2-dimensional] convolutions.
Note: Currently, n must be <= 3
"""
from .engine import Value

class Tensor:
    def __init__(self, x):
        """
        desc.: The input must be a tensor, with <= 3 dimensions
        input:
            x : list[Union(float, list[Union(float, list[float])])]
        """
        assert isinstance(x, list), 'The 1st dimension must be a list'
        if all(isinstance(y, (float, int, Value)) for y in x): # 1d
            self.shape= (len(x),)
            self.tensor= [e if isinstance(e, Value) else Value(e) for e in x]
        elif all(isinstance(y, list) for y in x):
            if all(isinstance(z, (float, int, Value)) for y in x for z in y): # 2d
                assert all(len(x[0]) == len(y) for y in x[1:]), 'The 2nd dimension contains lists of different lengths'
                self.shape= (len(x), len(x[0]))
                self.tensor= [ [e if isinstance(e, Value) else Value(e) for e in y] for y in x]
            elif all(isinstance(z, list) for y in x for z in y):
                if all(isinstance(u, (float, int, Value)) for y in x for z in y for u in z): # 3d
                    assert all(len(x[0]) == len(y) for y in x[1:]), 'The 2nd dimension contains lists of different lengths'
                    assert all(len(x[0][0]) == len(z) for y in x for z in y), 'The 3rd dimension contains lists of different lengths'
                    self.shape= (len(x), len(x[0]), len(x[0][0]))
                    self.tensor= [ [ [e if isinstance(e, Value) else Value(e) for e in z] for z in y] for y in x]
                else:
                    raise Exception('The 3rd dimension must consistently contain values of type float/int.')
            else:
                raise Exception('The 2nd dimension must consistently contain values of type: either float/int or list')
        else:
            raise Exception('The 1st dimension must consistently contain values of type: either float/int or list')

    def __len__(self):
        return self.shape[0]
    
    def __repr__(self): # pretty-printing
        return f'Dimension: {len(self.shape)}. Shape: {self.shape}'
        """# to print the details of the tensor-object
        print(f'Dimension: {len(self.shape)}. Shape: {self.shape}')
        if len(self.shape) == 1:
            return '['+' '.join(list(map(lambda e : str(e.val), self.tensor)))+']'
        elif len(self.shape) == 2:
            (rows, cols)= self.shape
            matrix= []
            for r in range(rows):
                matrix.append('['+' '.join(list(map(lambda e : str(e.val), self.tensor[r])))+']')
            return '['+',\n'.join(matrix)+']'
        elif len(self.shape) == 3:
            (depth, rows, cols)= self.shape
            tensor= []
            for d in range(depth):
                matrix= []
                for r in range(rows):
                    matrix.append('['+' '.join(list(map(lambda e : str(e.val), self.tensor[d][r])))+']')
                tensor.append('['+',\n'.join(matrix)+']')
            return '[\n'+',\n\n'.join(tensor)+'\n]'"""

    def __getitem__(self, idx):
        """
        desc.: to extract slices from the tensor
        """
        if len(self.shape) == 1: # 1d
            assert isinstance(idx, (int, slice)), "Index must be an instance of type: {int, slice}"
            vector= self.tensor[idx]
            # the class 'Tensor' expects [as input] a 'list', not an 'int'
            return Tensor([vector] if isinstance(idx, int) else vector)
        elif len(self.shape) == 2: # 2d
            assert isinstance(idx, tuple) and len(idx) == 2, "Index must be a tuple, of size 2"
            assert all(isinstance(i, (int, slice)) for i in idx), "Indices must be instances of type: {int, slice}"
            (i, j)= idx
            if isinstance(i, int):
                return Tensor([self.tensor[i][j]] if isinstance(j, int) else self.tensor[i][j])
            else:
                return Tensor([row[j] if isinstance(j, slice) else [row[j]] for row in self.tensor[i]])
        elif len(self.shape) == 3: # 3d
            assert isinstance(idx, tuple) and len(idx) == 3, "Index must be a tuple, of size 3"
            assert all(isinstance(i, (int, slice)) for i in idx), "Indices must be instances of type: {int, slice}"
            (i, j, k)= idx
            if isinstance(i, int):
                matrix= self.tensor[i]
                if isinstance(j, int):
                    row= matrix[j]
                    return Tensor([row[k]] if isinstance(k, int) else row[k])
                else:
                    return Tensor([row[k] if isinstance(k, slice) else [row[k]] for row in matrix[j]])
            else:
                tensor= []
                for matrix in self.tensor[i]:
                    if isinstance(j, int):
                        tensor.append([matrix[j][k]] if isinstance(k, slice) else [[matrix[j][k]]])
                    else:
                        tensor.append([row[k] if isinstance(k, slice) else [row[k]] for row in matrix[j]])
                return Tensor(tensor)
        else:
            print('Invalid index.', self)

    def flatten(self):

        """
        desc.: converts a 2/3d tensor to a 1d tensor [aka a vector]
        e.g.,
        dimension: (2, 3, 4) -> (2*3*4,)
        values:
            
            [
                [
                    [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11],
                ],
                [
                    [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23],
                ],
            ]

            is transformed to

            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        """

        # convert 2d to 3d; to reduce the code required to flatten the tensor
        if len(self.shape) == 2:
            self.tensor= [self.tensor] # 2d -> 3d
            self.shape= (1, *self.shape)

        # 3d -> 1d
        (d, r, c)= self.shape
        vector=[]
        for i in range(d):
            for j in range(r):
                vector.extend(self.tensor[i][j])

        self.tensor= vector
        self.shape = (d*r*c,)

    def reshape(self, new_dim):

        """
        reshapes a tensor [i.e., 1/2/3d] to 2/3d
        """

        assert isinstance(new_dim, tuple) and 2 <= len(new_dim) <= 3, "'new_dim' must be a tuple, of size 2 or 3"
        assert all(isinstance(x, int) and x > 0 for x in new_dim), "the values of 'new_dim' must be positive integers"

        # 1. 2/3d -> 1d
        if len(self.shape) > 1:
            self.flatten()

        # 2. 1d -> 3d
        # a. if 'new_dim' is 2d, convert it to 3d
        if len(new_dim) == 2:
            (depth, rows, cols)= (1, *new_dim)
        else:
            (depth, rows, cols)= new_dim

        # b. 1d -> 3d
        assert depth*rows*cols == self.shape[0], "len(vector) != depth*rows*cols"
        tensor=[]
        channel = rows*cols
        for i in range(depth):
            offset= channel*i
            matrix= []
            for j in range(rows):
                k= j*cols+offset
                matrix.append(self.tensor[k:k+cols])
            tensor.append(matrix)

        # 3. 2/3d
        self.tensor= tensor if len(new_dim) == 3 else tensor[0]
        self.shape = new_dim