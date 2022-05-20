import numpy as np
import matplotlib.pyplot as plt
from parent import print
from dezero import Variable
import dezero.functions as F

# 선형 회귀

np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)
# x, y = Variable(x), Variable(y)

W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))


def predict(x):
    y = F.matmul(x, W) + b
    return y


def mean_squared_error_simple(x0, x1):
    diff = x0 - x1
    # return F.sum(diff ** 2) / len(diff) # TODO: 오류 원인 밝혀내기
    return np.sum(diff ** 2) / len(diff)


lr = 0.1
iters = 1000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)
    # loss = mean_squared_error_simple(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data

    print('W', W, 'b', b, 'loss', loss)


plt.plot(x.data, y.data, 'o')
x = np.arange(0, 1, 0.01)
y_pred = x * W.data + b.data
plt.plot(x, y_pred.reshape(-1))
plt.get_current_fig_manager().window.showMaximized() # pip install pyqt5
plt.show()
