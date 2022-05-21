import numpy as np
import matplotlib.pyplot as plt
from parent import print
from dezero import Model
import dezero.layers as L
import dezero.functions as F
import dezero.models as M

# 계층을 모아두는 계층


# 데이터셋
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)


lr = 0.2
max_iter = 10000
hidden_size = 10


# 모델 정의
class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y


model = TwoLayerNet(hidden_size, 1)    
# model = M.MLP((10, 20, 10, 1))


# 학습 시작
for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data

    if i % 1000 == 0:
        print('iter', i, '| loss', loss)


x_test = np.arange(0, 1, 0.01).reshape(-1, 1)
y_pred = model(x_test)
plt.plot(x_test, y_pred.data)
plt.plot(x.data, y.data, 'o')
plt.get_current_fig_manager().window.showMaximized()
plt.show()
