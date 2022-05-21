import numpy as np
import matplotlib.pyplot as plt
from parent import print
import dezero.functions as F
import dezero.layers as L

# 매개변수를 모아두는 계층


# 데이터셋
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)


l1 = L.Linear(10) # 출력 크기 지정
l2 = L.Linear(1)


# 신경망 추론
def predict(x):
    y = l1(x)
    y = F.sigmoid(y)
    y = l2(y)
    return y


lr = 0.2
iters = 10000

# 신경망 학습
for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    l1.cleargrads()
    l2.cleargrads()
    loss.backward()

    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data

    if i % 1000 == 0:
        print('loss', loss)


x_test = np.arange(0, 1, 0.01).reshape(-1, 1)
y_pred = predict(x_test)
plt.plot(x_test, y_pred.data)
plt.plot(x.data, y.data, 'o')
plt.get_current_fig_manager().window.showMaximized()
plt.show()
