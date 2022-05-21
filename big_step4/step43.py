import numpy as np
import matplotlib.pyplot as plt
from parent import print
from dezero.core import as_variable
from dezero import Variable
import dezero.functions as F

# 신경망


# 데이터셋
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)


# 가중치 초기화
I, H, O = 1, 10, 1
W1 = Variable(0.01 * np.random.randn(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.randn(H, O))
b2 = Variable(np.zeros(O))


def linear_simple(x, W, b=None):
    t = F.matmul(x, W)
    if b is None:
        return t

    y = t + b
    t.data = None
    return y


def sigmoid_simple(x):
    x = as_variable(x)
    y = 1 / (1 + F.exp(-x))
    return y


# 신경망 추론
def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    return y


lr = 0.2
iters = 10000

# 신경망 학습
for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()

    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data
    
    if i % 1000 == 0:
        print('loss', loss)


x_test = np.arange(0, 1, 0.01).reshape(-1, 1)
y_pred = predict(x_test)
plt.plot(x_test, y_pred.data)
plt.plot(x.data, y.data, 'o')
plt.get_current_fig_manager().window.showMaximized()
plt.show()
