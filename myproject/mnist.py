import os
import time
import matplotlib.pyplot as plt

from parent import print
import dezero.datasets
from dezero import DataLoader, optimizers, no_grad, cuda
from dezero.models import Model, MLP
import dezero.functions as F
import dezero.layers as L
from dezero.transforms import Compose, Flatten, ToFloat, Normalize
from rich.progress import track


from plot import (
    animate_training_info, 
    plot_training_info,
)

# 모델 정의
class ConvNet(Model):
    def __init__(self):
        super().__init__()
        # self.conv1 = L.Conv2d(16, kernel_size=3, stride=1, pad=1)
        # self.conv2 = L.Conv2d(16, kernel_size=3, stride=1, pad=1)
        # self.conv3 = L.Conv2d(32, kernel_size=3, stride=1, pad=1)
        # self.conv4 = L.Conv2d(32, kernel_size=3, stride=1, pad=2)
        self.fc5 = L.Linear(1)
        self.fc6 = L.Linear(1)
        self.fc7 = L.Linear(10)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.pooling(x, 2, 2)

        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = F.pooling(x, 2, 2)

        # x = F.reshape(x, (x.shape[0], -1))
        x = F.dropout(F.relu(self.fc5(x)))
        x = F.dropout(F.relu(self.fc6(x)))
        x = self.fc7(x)
        return x


# 하이퍼파라미터 설정
max_epoch = 10
batch_size = 100
lr = 0.1
save_path = 'myproject/mnist_model.npz'


# 데이터 읽기, 모델, 옵티마이저 생성
train_set = dezero.datasets.MNIST(train=True, transform=Compose([Flatten(), ToFloat(), Normalize(0., 255.)]))
test_set = dezero.datasets.MNIST(train=False, transform=Compose([Flatten(), ToFloat(), Normalize(0., 255.)]))
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size)
print('학습 데이터 불러오기 완료!')

model = ConvNet()
optimizer = optimizers.MomentumSGD(lr).setup(model)


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
    train_loader_tracking = track(
        train_loader, total=train_loader.max_iter, 
        description=f'epoch {epoch+1} / {max_epoch}'
    )
    start = time.perf_counter()
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader_tracking:
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

    print('epoch {} ({:.2f}sec)'.format(epoch+1, elapsed_time), end=' | ')
    print('train loss: {:.4f}, accuracy: {:.2f}%'.format(losses_train[-1], accs_train[-1]), end=' | ')
    
    sum_loss, sum_acc = 0, 0
    with no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)
            
            if epoch == max_epoch-1:
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
    video_time_sec=10, save=True, save_path='myproject/MNIST_train_info.gif'
)
plt.get_current_fig_manager().window.showMaximized()
plt.show()
