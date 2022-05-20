import numpy as np
from parent import print
from dezero import Variable
import dezero.functions as F

# 행렬의 곱

print('matmul', style='bold yellow')

x = Variable(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
W = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.matmul(x, W)
y.backward()

print('x', x, sep='\n')
print('W', W, sep='\n')
print('y', y, sep='\n')
print('W.T', W.T, sep='\n')
print('x.grad', x.grad, sep='\n')
print('W.grad', W.grad, sep='\n')
