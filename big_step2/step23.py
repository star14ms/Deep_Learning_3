
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np
from dezero import Variable
from rich_console import print

# 패키지로 정리

print(os.path.dirname(__file__))
print(os.path.join(os.path.dirname(__file__), '..'))


x = Variable(np.array(1.0))
y = (x + 3) ** 2
y.backward()

print('x', x)
print('x.grad', x.grad)