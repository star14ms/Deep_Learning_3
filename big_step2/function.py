import numpy as np
import weakref
from rich import print


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


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

        
class Add(Function):
    def forward(self, *xs):
        y = np.sum(xs)
        self.len_xs = len(xs)
        return (y,)

    def backward(self, gy):
        return (gy,)*self.len_xs


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        self.x0, self.x1 = x0, x1
        return (y,)

    def backward(self, gy):
        return self.x1*gy, self.x0*gy


def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def add(*xs):
    return Add()(*xs)

def mul(*xs):
    return Mul()(*xs)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


if __name__ == '__main__':
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(2.0))
    x2 = Variable(np.array(3.0))

    def print_grad_and_reset(y, *xs):
        y.backward()

        print('y', y.data)
        for i, x in enumerate(xs):
            print(f'x{i}.grad', x.grad)
        print()
    
        x0.cleargrad()
        x1.cleargrad()
        x2.cleargrad()

    y = add(x0, x1, x2)
    print_grad_and_reset(y, x0, x1, x2)

    ################################

    y = mul(mul(x0, x1), x2)
    print_grad_and_reset(y, x0, x1, x2)

    ################################

    y = mul(add(x0, x1), x1)
    print_grad_and_reset(y, x0, x1, x2)
    
    ################################
    
    x0 = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])
    x1 = np.array([
        [5.0, 6.0],
        [7.0, 8.0]
    ])
    x2 = np.array([
        [9.0, 10.0],
        [11.0, 12.0]
    ])

    print('x0*x1*x2\n' + str(x0*x1*x2))
