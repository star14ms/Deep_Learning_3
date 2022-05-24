import os
import time
import matplotlib.pyplot as plt

from parent import print
import dezero.datasets
from dezero import DataLoader, optimizers, no_grad, cuda
from dezero.models import MLP
import dezero.functions as F

from plot import (
    animate_training_info, 
    plot_training_info,
)


# MNIST 학습


# 하이퍼파라미터 설정
max_epoch = 10
batch_size = 100
hidden_size = 1000
lr = 1.0
save_path = 'big_step5/my_mlp.npz'


# 데이터 읽기, 모델, 옵티마이저 생성
train_set = dezero.datasets.MNIST(train=True)
test_set = dezero.datasets.MNIST(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size)
print('학습 데이터 불러오기 완료!')

model = MLP((hidden_size, 10), activation=F.relu)
optimizer = optimizers.SGD(lr).setup(model)


if os.path.exists(save_path):
    model.load_weights(save_path)
    print('모델 불러오기 완료!')

# GPU 모드
if dezero.cuda.gpu_enable:
    train_loader.to_gpu()
    test_loader.to_gpu()
    model.to_gpu()


losses_train, accs_train = [], []
losses_test, accs_test = [], []
wrong_idxs, ys, ts = [], [], []

print('학습 시작!')
for epoch in range(max_epoch):
    start = time.perf_counter()
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    losses_train.append(sum_loss / len(train_set))
    accs_train.append(sum_acc / len(train_set) * 100)
    elapsed_time = time.perf_counter() - start

    print('epoch {} {:.2f}[sec]'.format(epoch+1, elapsed_time), end=' | ')
    print('train loss: {:.4f}, accuracy: {:.2f}%'.format(losses_train[-1], accs_train[-1]), end=' | ')
    
    sum_loss, sum_acc = 0, 0
    with no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

            wrong_idxs.extend(cuda.as_numpy(y.data.argmax(1) != t))
            ys.extend(cuda.as_numpy(y.data))
            ts.extend(t)

    losses_test.append(sum_loss / len(test_set))
    accs_test.append(sum_acc / len(test_set) * 100)
    print('test loss: {:.4f}, accuracy: {:.2f}%'.format(losses_test[-1], accs_test[-1]))


model.save_weights(save_path.replace())


# 학습 정보 시각화
info = losses_train, losses_test, accs_train, accs_test

figure, axes = plt.subplots(2, 1)
plot_training_info(axes, *info)
figure.canvas.manager.window.showMaximized()
plt.show()


# 애니메이션
animation = animate_training_info(*info,
    video_time_sec=10, save=True, save_path='big_step5/MNIST_train_info.gif'
)
plt.get_current_fig_manager().window.showMaximized()
plt.show()
