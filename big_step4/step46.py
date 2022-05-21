import numpy as np
import matplotlib.pyplot as plt
from parent import print
import dezero.functions as F
import dezero.models as M
from dezero import optimizers

# Optimizer로 수행하는 매개변수 갱신


# 데이터셋
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)


lr = 0.2
max_iter = 10000
hidden_size = 10


model = M.MLP((10, 20, 10, 1))
optimizer = optimizers.MomentumSGD(lr).setup(model) # SGD, MomentumSGD


# 학습 시작
for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()

    if i % 1000 == 0:
        print('iter', i, '| loss', loss)


x_test = np.arange(0, 1, 0.01).reshape(-1, 1)
y_pred = model(x_test)
plt.plot(x_test, y_pred.data)
plt.plot(x.data, y.data, 'o')
plt.get_current_fig_manager().window.showMaximized()
plt.show()
