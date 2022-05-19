
# 함수 최적화

def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (1 - x0) ** 2
    return y


if __name__ == '__main__':
    import numpy as np
    from parent import print
    from dezero import Variable
    import matplotlib.pyplot as plt
    from matplotlib import ticker, cm


    x0 = Variable(np.array(0.0))
    x1 = Variable(np.array(2.0))
    
    lr = 0.001
    iters = 1000
    
    info_x0 = []
    info_x1 = []

    for i in range(iters):
        print('iter', i, x0, x1)
    
        y = rosenbrock(x0, x1)
        x0.cleargrad()
        x1.cleargrad()
        y.backward()

        info_x0.append(x0.data.copy())
        info_x1.append(x1.data.copy())
    
        x0.data -= lr * x0.grad
        x1.data -= lr * x1.grad
    
    
    # 그래프로 시각화
    x = np.arange(-2, 2, 0.01)
    y = np.arange(-1, 3, 0.01)
    
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)

    plt.figure()
    CS = plt.contourf(
        X, Y, Z, 
        levels=[10**i for i in range(-2, 4)], 
        locator=ticker.LogLocator(),
        cmap=cm.PuBu_r # cm.PuBu_r 'seismic'
    )
    plt.clabel(CS, inline=2, fontsize=20, colors='black')
    plt.colorbar()

    plt.plot(info_x0, info_x1, color='tab:orange')
    plt.plot(info_x0, info_x1, 'o', color='tab:red')
    plt.plot(1, 1, '*', color='tab:blue')

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized() # pip install pyqt5
    plt.show()
