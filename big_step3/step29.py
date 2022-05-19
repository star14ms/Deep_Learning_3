import numpy as np
from parent import print
from dezero import Variable
import matplotlib.pyplot as plt

# 뉴턴 방법으로 푸는 최적화 (수동 계산)

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y

def gx2(x):
    return 12 * x ** 2 - 4


x = Variable(np.array(2.0))
iters = 10

info_x = []
info_y = []
for i in range(iters):
    print('iter', i, x)

    y = f(x)
    x.cleargrad()
    y.backward()

    info_x.append(x.data.copy())
    info_y.append(y.data)

    x.data -= x.grad / gx2(x.data)


# 그래프로 시각화
x = np.arange(-2.1, 2.1, 0.01)
y = f(x)
plt.plot(x, y, info_x, info_y)
plt.plot(info_x, info_y, 'o', color='tab:orange')
plt.show()
