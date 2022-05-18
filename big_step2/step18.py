import numpy as np
from function import as_array
import weakref
from console import print

import contextlib


class Config:
    enable_backprop = True


@contextlib.contextmanager
def config_test():
    print('start')
    try:
        yield
    finally:
        print('done')


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config('enable_backprop', False)


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

    def backward(self, retain_grad=False):
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
        
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    def cleargrad(self):
        self.grad = None


class Function():
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        
        if Config.enable_backprop: # 역전파 활성 모드
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


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return (y,)

    def backward(self, gy):
        return gy, gy


def add(x, y):
    return Add()(x, y)


if __name__ == '__main__':
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(1.0))
    
    def backprop_test(n):
        try:
            t = add(x0, x1)
            y = add(x0, t)
            y.backward()
            print('y.grad, t.grad', y.grad, t.grad)
            print('x0.grad, x1.grad', x0.grad, x1.grad)
        
            x0.cleargrad()
            x1.cleargrad()
            print(n, 'success')

        except Exception as e:
            print(n, repr(e))
        finally:
            print()

    backprop_test(1)
    Config.enable_backprop = False
    backprop_test(2)
    
    with no_grad():
        backprop_test(3)
    
    with config_test():
        print('process...')