import numpy as np
from parent import print
from dezero import Variable
from dezero.utils import plot_dot_graph
from dezero.core import vname_auto_gen


x = Variable(np.array(2), name='x = 2')

with vname_auto_gen():
    y = x ** 2
    y.backward(create_graph=True)
    print("y'", x.grad)

gx = x.grad
y.name = f'y = {y.data}'
gx.name = f'gx = {gx.data}'
x.cleargrad()

z = gx ** 3 + y
z.backward()
print("z'", x.grad)


# 계산 그래프 그리기
gx.name = f'gx = {gx.data}'
plot_dot_graph(gx, verbose=False, to_file='big_step3/graph/step36.png')

# z.name = f'z = {z.data}'
# plot_dot_graph(z, verbose=False, to_file='big_step3/graph/step36.png')