import numpy as np
from parent import print
from dezero import Variable
import dezero.functions as F

# 합계 함수

print('sum', style='bold yellow')

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.sum(x)
y.backward()
print('x', x, sep='\n')
print('y', y, sep='\n')
print('x.grad', x.grad, sep='\n')

y = x.sum(axis=1, keepdims=True)
print('x.sum(axis=1, keepdims=True)', y, sep='\n')
