import numpy as np
from parent import print
from dezero import Variable
import dezero.functions as F

# 형상 변환 함수

print('reshape', style='bold yellow')

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.reshape(x, (6,))
y.backward()
print('x', x, sep='\n')
print('y', y, sep='\n')
print('x.grad', x.grad, sep='\n')

y = x.reshape((2, 3))
print('y', y, sep='\n')

y = x.reshape(2, 3)
print('y', y, '\n', sep='\n')

################################################################

print('transpose', style='bold yellow')

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.transpose(x)
y.backward(retain_grad=True)
print('x', x, sep='\n')
print('y', y, sep='\n')
print('x.grad', x.grad, sep='\n')

y = x.transpose(1, 0)
print('x.transpose(1, 0)', y, sep='\n')

y = x.T
print('x.T', y, sep='\n')

################################################################

# # np.argsort
# A, B, C, D = 1, 2, 3, 4
# axes = 2, 1, 3, 0
# x = np.random.randn(1, 2, 3, 4)

# y = x.transpose(axes)
# print(y.shape)

# inv_axes = np.argsort([ax % 4 for ax in [axes]])
# z = y.transpose(inv_axes)
# print('arg', [ax % 4 for ax in [axes]])
# print(inv_axes)
# print(z.shape)
