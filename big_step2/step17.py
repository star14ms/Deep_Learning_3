import numpy as np
from function import as_array

import weakref
import gc
from memory_profiler import profile
import time

from step16 import (
    Variable as Variable_old, 
    square as square_old, 
)
from rich.syntax import Syntax
from console import print


class Variable():
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self):
        if not self.grad:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)
    
        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx 

                if x.creator is not None:
                    add_func(x.creator)
        
    def cleargrad(self):
        self.grad = None


class Function():
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        
        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)
            
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]
    
        return outputs if len(outputs) > 1 else outputs[0]
        
    def forward(self, x):
        raise NotImplementedError

    def backward(self, gy):
        raise NotImplementedError


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def square(x):
    return Square()(x)


f1=open('big_step2/weakref.txt','w+')

@profile(stream=None) # stream=f1
def memory_profile(variable, square):
    for _ in range(10):
        x = variable(np.random.randn(100000))
        _ = square(square(square(x)))

@profile()
def test():
    pass


# 그래프 보기:  
# mprof run big_step2/step17to18.py
# mprof plot -o big_step2/image.png --backend agg
if __name__ == '__main__':
    a = Variable(np.array(2.0))
    b = weakref.ref(a)
    code = \
"""\
a = Variable(np.array(2.0))
b = weakref.ref(a)\
"""
    print('', Syntax(code, 'python', line_numbers=True), '')
    print('b', b)
    print('b().data', b().data)
    
    a = None
    print('', Syntax('a = None', 'python', line_numbers=True), '')
    print('b', b, '\n')

    test()
    gc.collect()

    time.sleep(1)
    memory_profile(Variable, square)
    gc.collect()

    time.sleep(1)
    memory_profile(Variable_old, square_old)
    gc.collect()
