import numpy as np
from parent import print
from dezero import Variable
from dezero.utils import plot_dot_graph
from big_step2.step24 import goldstein


if __name__ == '__main__':
    x = np.array(1.0)

    x = Variable(np.array(1), name='x = 1')
    y = Variable(np.array(1), name='y = 1')
    z = goldstein(x, y)
    z.backward()
    
    z.name = f'z = {z.data}'

    print(f'{goldstein.__name__}(1,1)', z)
    print('x.grad', x.grad)
    print('y.grad', y.grad, '\n')
    
    plot_dot_graph(z, verbose=False, to_file='big_step3/graph/goldstein.png')
    plot_dot_graph(z, verbose=True, to_file='big_step3/graph/goldstein_verbose.png')
