import numpy as np
import math
from parent import print
from dezero.utils import plot_dot_graph
from dezero import Variable, Function


# 테일러 급수 미분

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx


def sin(x):
    return Sin()(x)


def my_sin(x, threshold=0.0001) -> Variable:
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        
        if abs(t.data) < threshold:
            break
    
    return y


if __name__ == '__main__':

    for threshold in [0.0001, 1e-150]:
        x = Variable(np.array(np.pi/4), name='x')
        y = my_sin(x, threshold=threshold)
        y.backward()

        y.name = 'y'
        
        print('threshold', threshold)
        print('y.data', y.data)
        print('x.grad', x.grad)
        print('1/np.sqrt(2)', 1/np.sqrt(2), '\n')

        plot_dot_graph(y, verbose=False, to_file=f'big_step3/graph/step27_sin_taylor_th_{threshold}.png')
        
        x.cleargrad()
