import numpy as np
from parent import print
from dezero import Variable
import dezero.functions as F

# 텐서를 다루다

x = Variable(np.array(1.0))
y = F.sin(x)

print('x', x)
print('sin(x)', y, '\n')


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.sin(x)

print('x', x)
print('sin(x)', y, '\n')


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
c = Variable(np.array([[10, 20, 30], [40, 50, 60]]))
t = x + c
y = F.sum(t)
y.backward(retain_grad=True)

print('x', x)
print('x + c', y, '\n')

print('y.grad', y.grad)
print('t.grad', t.grad)
print('c.grad', c.grad)
print('x.grad', x.grad, '\n')
