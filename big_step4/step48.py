import numpy as np
import matplotlib.pyplot as plt
import math

from parent import print
import dezero.datasets
from dezero import optimizers
from dezero.models import MLP
import dezero.functions as F

from plot import plot_decision_boundary


# 다중 클래스 분류


# 하이퍼파라미터 설정
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

# 데이터 읽기, 모델, 옵티마이저 생성
x, t = dezero.datasets.get_spiral(train=True)
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(x)
max_iter = math.ceil(data_size / batch_size)


fig = plt.figure()

for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        batch_index = index[i * batch_size:(i + 1) * batch_size]
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        
        model.cleargrads()
        loss.backward()
        optimizer.update()
        
        sum_loss += float(loss.data) * batch_size

    avg_loss = sum_loss / data_size
    print('epoch %d, loss %.2f' % (epoch+1, avg_loss))
    

plot_decision_boundary(model, x, t, title='Spiral Dataset Classfication')
plt.show()
