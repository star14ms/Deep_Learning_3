import numpy as np
import matplotlib.pyplot as plt

from parent import print
import dezero.datasets
from dezero import DataLoader, optimizers, no_grad
from dezero.models import MLP
import dezero.functions as F

from plot import plot_decision_boundary, animate_training_info, plot_training_info
from matplotlib.animation import FFMpegWriter # FFMpeg 설치 필요


# 미니배치를 뽑아주는 DataLoader


# 하이퍼파라미터 설정
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0
save_animation = False # 결정 경계 변화 과정 애니메이션으로 저장


# 데이터 읽기, 모델, 옵티마이저 생성
train_set = dezero.datasets.Spiral(train=True)
test_set = dezero.datasets.Spiral(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)


losses_train, accs_train = [], []
losses_test, accs_test = [], []


fig = plt.figure()


# 결정 경계 변화 과정 애니메이션으로 저장
if save_animation:
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=5, metadata=metadata)
    writer.setup(fig, 'big_step4/spiral_decision_boundary.mp4', dpi=100)


for epoch in range(max_epoch):
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

    print('epoch {}'.format(epoch+1), end=' | ')
    print('train loss: {:.4f}, accuracy: {:.2f}%'.format(losses_train[-1], accs_train[-1]), end=' | ')
    
    sum_loss, sum_acc = 0, 0
    with no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    losses_test.append(sum_loss / len(test_set))
    accs_test.append(sum_acc / len(test_set) * 100)
    print('test loss: {:.4f}, accuracy: {:.2f}%'.format(losses_test[-1], accs_test[-1]))
    
    # 애니메이션 프레임 추가
    if save_animation:
        fig.clear()
        axes = plt.axes()
        plot_decision_boundary(axes, model, test_set, title='Spiral Dataset Classfication')
        writer.grab_frame()


# 결정 경계 시각화
if save_animation:
    writer.finish()
else:
    axes = plt.axes()
    plot_decision_boundary(axes, model, test_set, title='Spiral Dataset Classfication')
plt.show()


# 학습 정보 시각화
info = losses_train, losses_test, accs_train, accs_test

figure, axes = plt.subplots(2, 1)
plot_training_info(axes, *info)
figure.canvas.manager.window.showMaximized()
plt.show()


# 애니메이션
animation = animate_training_info(*info,
    video_time_sec=10, save=True, save_path='big_step4/sprial_train_info.gif'
)
plt.get_current_fig_manager().window.showMaximized()
plt.show()
