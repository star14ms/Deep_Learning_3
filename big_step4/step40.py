import numpy as np
from parent import print
from dezero import Variable
import dezero.functions as F

# 브로드캐스트 함수

print('broadcast', style='bold yellow')

x0 = Variable(np.array([1, 2, 3]))
x1 = Variable(np.array([10]))
y = x0 + x1
y.backward()

print('x0', x0, sep='\n')
print('x1', x1, sep='\n')
print('y', y, sep='\n')
print('x0.grad', x0.grad, sep='\n')
print('x1.grad', x1.grad, sep='\n')
