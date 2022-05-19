import numpy as np
from parent import print
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F


x = Variable(np.array(1.0), name='x')
y = F.tanh(x)
y.name = 'y'
y.backward(create_graph=True)
print("y'", x.grad)

iters = 5

for i in range(iters):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)
    gx.name = 'gx' + str(iters+2)
    print('y'+"'"*(i+2), gx)


# 계산 그래프 그리기
gx = x.grad
gx.name = 'gx' + str(iters+1)
plot_dot_graph(gx, verbose=False, to_file='big_step3/graph/step35_tanh.png')