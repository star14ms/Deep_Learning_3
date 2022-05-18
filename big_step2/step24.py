import numpy as np
from parent import print
from dezero import Variable
from dezero.core_simple import mul

import time
import sys


def sphere(x, y):
    z = x ** 2 + y ** 2
    return z

def sphere_general(*xs):
    z = sum(list(map(mul, xs, xs)))
    return z

# f(x,y)=0.26\left(x^{2}+y^{2}\right)-0.48xy
def matyas(x, y): # 마차시
    z = 0.26 * (x**2 + y**2) - 0.48*x*y
    return z


def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) \
      * (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z


xs = (i for i in range(1, 100+1))
print(sys.getsizeof(xs), 'bytes')
xs = [Variable(np.array(x)) for x in xs]
print(sys.getsizeof(xs), 'bytes')

z = sphere_general(*xs)
z.backward()

print(f'{sphere_general.__name__}({xs})', z)

for dim, x in enumerate(xs):
  print(f'{dim+1}th_dim.grad', xs[dim].grad, end="\r")
  time.sleep(0.025)
print('\n')


x = Variable(np.array(1.0))
y = Variable(np.array(1.0))

for func in [matyas, goldstein]:
    z = func(x, y)
    z.backward()
    
    print(f'{func.__name__}(1,1)', z)
    print('x.grad', x.grad)
    print('y.grad', y.grad, '\n')
    
    x.cleargrad()
    y.cleargrad()

