import numpy as np
from function import as_array
from parent import print
np.set_printoptions(precision=4)


class Variable():
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if not self.grad:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs] ### 수정된 구간
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx 

                if x.creator is not None:
                    funcs.append(x.creator) ### 여기까지

    def cleargrad(self):
        self.grad = None


class Function():
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
            
        self.inputs = inputs
        self.outputs = outputs

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


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return (y,)

    def backward(self, gy):
        return gy, gy


def square(x):
    return Square()(x)

def add(x, y):
    return Add()(x, y)


if __name__ == '__main__':
    x0 = Variable(np.array(2.0))
    x1 = Variable(np.array(3.0))
    y = add(x0, x1)
    print('y.data', y.data, '\n')
    
    x0 = Variable(np.array([1.0, 2.0, 3.0]))
    x1 = Variable(np.array([1.0, 2.0, 3.0]))
    z = add(square(x0), square(x1))
    z.backward()
    print('z.data', z.data)
    print('x0.grad', x0.grad)
    print('x1.grad', x1.grad, '\n')
    
    
    x = Variable(np.array(3.0))
    y = add(x, x)
    y.backward()
    print('y', y.data)
    print('x.grad', x.grad, '\n')
    
    x.cleargrad()
    
    y = add(add(x, x), x)
    y.backward()
    print('x.grad', x.grad, '\n')
