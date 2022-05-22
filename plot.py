import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
import matplotlib.ticker as ticker

from dezero import Dataset
from dezero.functions import softmax
import math


def plot_spiral_data(axes, x, t):
    x_0 = x[np.where(t == 0)]
    x_1 = x[np.where(t == 1)]
    x_2 = x[np.where(t == 2)]
    axes.plot(x_0[:,0], x_0[:,1], color='orange', marker='o', linestyle='', ms=12, label='class_A')
    axes.plot(x_1[:,0], x_1[:,1], 'xb', ms=12, mew=3, label='class_B')
    axes.plot(x_2[:,0], x_2[:,1], '^g', ms=12, label='class_C')
    axes.legend()

    return axes


def plot_decision_boundary(axes, model, *data, h=.02, title='', cmap_light=None, cmap_bold=None):
    if isinstance(data[0], Dataset):
        x = np.array([example[0] for example in data[0]])
        t = np.array([example[1] for example in data[0]])
    else:
        x, t = data
     
    if cmap_light is None:
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    if cmap_bold is None:
        # cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
        cmap_bold = ListedColormap(['orange', 'blue', 'green'])

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    extend = 0.1
    x_min, x_max = x[:, 0].min() - extend, x[:, 0].max() + extend
    y_min, y_max = x[:, 1].min() - extend, x[:, 1].max() + extend
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h), # h: step size in the mesh
        np.arange(y_min, y_max, h)
    )
    
    y = model(np.c_[xx.ravel(), yy.ravel()])
    pred = y.data.argmax(axis=1).reshape(xx.shape)

    # Put the result into a color plot
    axes.pcolormesh(xx, yy, pred, cmap=cmap_light)
    axes.set_title(title)
    # axes.scatter(x[:, 0], x[:, 1], c=t, cmap=cmap_bold)
    axes = plot_spiral_data(axes, x, t)
    return axes


def plot_training_info(axes, losses_train=[], losses_test=[], accs_train=[], accs_test=[], 
    blank=False
):
    len_data = len(losses_test)
    
    if blank:
        line_train_losses, = axes[0].plot([], [], label='train loss')
        line_test_losses, = axes[0].plot([], [], label='test loss')
        line_train_accs, = axes[1].plot([], [], label='train acc')
        line_test_accs, = axes[1].plot([], [], label='test acc')
    else:
        axes[0].plot(np.arange(1, len_data+1), losses_train, label='train loss')
        axes[0].plot(np.arange(1, len_data+1), losses_test, label='test loss')
        axes[1].plot(np.arange(1, len_data+1), accs_train, label='train acc')
        axes[1].plot(np.arange(1, len_data+1), accs_test, label='test acc')
    
    fig = axes[0]
    fig.set_xlim(1, len_data)
    fig.set_ylim(0, max(losses_test))
    fig.set_xlabel('Epoch')
    fig.set_ylabel('Loss')
    fig.legend(loc='upper right')
    
    fig = axes[1]
    fig.set_xlim(1, len_data)
    fig.set_ylim(min(accs_train)-1, 100)
    fig.set_xlabel('Epoch')
    fig.set_ylabel('Accuracy')
    fig.legend(loc='lower right')
    
    if blank:
        return {
            'line_train_losses': line_train_losses,
            'line_train_accs': line_train_accs,
            'line_test_losses': line_test_losses,
            'line_test_accs': line_test_accs,
        }
    else:
        return None


def animate_training_info(losses_train, losses_test, accs_train, accs_test, 
    video_time_sec=10, save=False, save_path='train_sprial_dataset_250ms.gif'
):
    figure, axes = plt.subplots(2, 1)
    figure.set_size_inches(12.8, 7.6)

    len_data = len(accs_test)
    line_dict = plot_training_info(axes, losses_train, losses_test, accs_train, accs_test, blank=True)

    n_frame_skip = 0
    interval = 0
    while interval < 100: # 최소 0.1초에 한 번 업데이트
        n_frame_skip += 1
        interval = math.ceil(video_time_sec*1000 / len_data * n_frame_skip)

    def animate(i):
        until = n_frame_skip*i+1 if n_frame_skip*i+1 < len_data else len_data
        x = np.arange(1, until+1)
        line_dict['line_train_losses'].set_data(x, losses_train[:until])
        line_dict['line_test_losses'].set_data(x, losses_test[:until])
        line_dict['line_train_accs'].set_data(x, accs_train[:until])
        line_dict['line_test_accs'].set_data(x, accs_test[:until])

    anim = FuncAnimation(
        figure, animate, frames=len_data, 
        interval=interval, repeat=True, repeat_delay=3000,
    )
    
    if save:
        anim.save(save_path, writer='imagemagick')

    return anim


def imgs_show(img, nx='auto', filter_show=False, margin=3, scale=10, title_info=[], text_info=[], dark_mode=True, 
    adjust={'l':0, 'r':1, 'b':0.02, 't':0.98, 'hs':0.05, 'ws':0.02}):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    if len(img.shape) == 3:
        img = img.reshape(-1, *img.shape)

    FN, C, _, _ = img.shape
    imgs_num = FN if not filter_show else C
    if imgs_num == 1:
        nx = 1
    elif nx == 'auto': 
        nx = math.ceil(math.sqrt(imgs_num*108/192)/108*192)
        # print(nx)
    ny = int(np.ceil(imgs_num / nx))
    l, r, b, t, hs, ws = adjust['l'], adjust['r'], adjust['b'], adjust['t'], adjust['hs'], adjust['ws']

    fig = plt.figure()
    fig.subplots_adjust(left=l, right=r, bottom=b, top=t, hspace=hs, wspace=ws)
    cmap='gray' if dark_mode else plt.cm.gray_r

    for i in range(imgs_num):
        ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
        if title_info!=[]:
            plt.title(f'\n{title_info[i]}' if title_info!='' else f'{i+1}')
        if text_info!=[]:
            info = text_info[i].split(' | ')
            ax.text(0, 0, info[0], ha="left", va="top", fontsize=16, color='Green' if info[0] in info[-1] else 'Red')
            fig.canvas.draw()
            ax.text(18, 25, info[-1], ha="left", va="top", fontsize=16, color='Green' if info[0] in info[-1] else 'Red')
            fig.canvas.draw()
            
        ax.imshow(img[i, 0] if not filter_show else img[0, i], cmap=cmap, interpolation='nearest')

    plt.show()
        

def show_wrong_answers_info(wrong_xs, wrong_ys, dark_mode=True, title='', title_info=[], text_info=[]):
    len_wrongs = len(wrong_ys)
    cmap='gray' if dark_mode else plt.cm.gray_r
    size = 4 if len_wrongs != 1 else 2
    bar1_idxs = [2, 4, 10, 12] if len_wrongs != 1 else [2]
    bar2_idxs = [6, 8, 14, 16] if len_wrongs != 1 else [4]

    plots = 0
    for i, wrong_x, wrong_y in zip(range(len_wrongs), wrong_xs, wrong_ys):
        if plots == 0: 
            fig = plt.figure()
            fig.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.92, hspace=0.25, wspace=0.25)
        plots += 1

        plt.rcParams["font.size"] = 20
        ax = fig.add_subplot(size//2, size, 2*plots-1, xticks=[], yticks=[])
        if title_info!=[]:
            info = title_info[i] if title_info!='idx' else i+1
            if title=='':
                plt.title(f'{i+1}/{len_wrongs}\n{info}')
            else:
                plt.title(f'{i+1}/{len_wrongs}\n{title} ({info})') ### plot() 후에 나와야 함

        if text_info!=[]:
            ax.text(0, 0, text_info[i], ha="left", va="top", color='white')
            fig.canvas.draw()

        ax.imshow(wrong_x[0], cmap=cmap, interpolation='nearest')

        plt.rcParams["font.size"] = 11
        x = np.arange(10)
        y = wrong_y
        ax = fig.add_subplot(size, size, bar1_idxs[plots-1], xticks=x, yticks=np.round(sorted(y), 1), xlabel='손글씨 숫자 예측 | 위: 점수 | 아래: 확률(%)')
        ax.bar(x, y)
        
        y = softmax(wrong_y, axis=0)*100
        yticks = [y_tick.data for y_tick in sorted(y)[8:]]
        ax = fig.add_subplot(size, size, bar2_idxs[plots-1], xticks=x, yticks=yticks, ylim=(0, 100))
        ax.bar(x, y.data)

        
        if (i+1)%4 == 0 or i == len_wrongs-1: 
            plots = 0
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.pause(0.01)
            plt.show()


def set_font(font_path):
    import matplotlib.font_manager as fm

    font_name = fm.FontProperties(fname=font_path, size=50).get_name()
    plt.rc('font', family=font_name)


def print_font_list():
    from matplotlib import font_manager as fm
    
    font_list = fm.findSystemFonts() 
    font_list.sort()
    fnames = []
    for fpath in font_list:
        #폰트 파일의 경로를 사용하여 폰트 속성 객체 가져오기
        fp=fm.FontProperties(fname=fpath)
        
        # 폰트 속성을 통해 파이썬에 설정해야 하는 폰트 이름 조회 
        font_name=fp.get_name() 
        
        fnames.append(font_name)
        
    for idx, fname in enumerate(fnames):
        print(str(idx).ljust(4), fname)
     
    input()


if __name__ == '__main__':
    print_font_list()
    set_font('주아체.ttf')