import numpy as np
from function import as_array
import weakref
from config import Config
from parent import print

# 변수 사용성 개선

class Variable():
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))

        self.data = data
        self.name = name
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

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self):
        return self.data.shape

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'Variable(None)'
        else:
            p = str(self.data).replace('\n', '\n'+ ' '*9)
            return f'Variable({p})'

    def __mul__(self, other):
        return mul(self, other)


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


if __name__ == '__main__':
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]), name='똥')
    
    print('x.name', x.name)
    print('x.shape', x.shape)
    print('x.ndim', x.ndim)
    print('x.size', x.size)
    print('x.dtype', x.dtype, '\n')
    print('len(x)', len(x))
    print(x, '\n')
