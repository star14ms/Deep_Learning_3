import numpy as np
from parent import print
import dezero.functions as F
from dezero import Variable, as_variable
from dezero.models import MLP
from dezero.cuda import gpu_enable

# 소프트맥스 함수와 교차 엔트로피 오차

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
indices = np.array([0, 0, 1])
y = F.get_item(x, indices)
print('get_item([0, 0, 1])', y, sep='\n')

print('x[1]', x[1])
print('x[:,2]', x[:,2])


def softmax1d(x):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y


model = MLP((10, 3))

x = Variable(np.array([[0.2, -0.4]]))
y = model(x)
p = softmax1d(y)
print('y', y)
print('p', p, '\n')


# 교차 엔트로피 오차
print('GPU', gpu_enable)
x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t = np.array([2, 0, 1, 0])
y = model(x)
loss = F.softmax_cross_entropy(y, t)
print('loss', loss)
