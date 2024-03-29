import numpy as np
import weakref
import contextlib

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import dezero


class Config:
    enable_backprop = True
    train = True
    Variable_name_auto_gen = False


    
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


def test_mode():
    return using_config('train', False)


def vname_auto_gen():
    return using_config('Variable_name_auto_gen', True)


# =============================================================================
# Variable / Function
# =============================================================================
try:
    import cupy
    array_types = (np.ndarray, cupy.ndarray)
except ValueError:
    array_types = (np.ndarray)

class Variable():
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, array_types):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))

        self.data = data
        if Config.Variable_name_auto_gen and name is None and data.ndim == 0:
            scalar = data.tolist()
            if isinstance(scalar, int) or scalar.is_integer() or len(str(scalar)) < 2+6:
                self.name = str(scalar)
            else:
                self.name = np.format_float_scientific(scalar, precision=2, exp_digits=2)
        else:
            self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False, create_graph=False):
        if not self.grad:
            xp = dezero.cuda.get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))

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
            
            with using_config('enable_backprop', create_graph):
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
    
    def unchain(self):
        self.creator = None

    def unchain_backward(self):
        if self.creator is not None:
            funcs = [self.creator]
            while funcs:
                f = funcs.pop()
                for x in f.inputs:
                    if x.creator is not None:
                        funcs.append(x.creator)
                        x.unchain()

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

    @property
    def T(self):
        return dezero.functions.transpose(self)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'Variable(None)'
        else:
            p = str(self.data).replace('\n', '\n'+ ' '*9)
            return f'Variable({p})'
    
    def __eq__(self, other): # 항등 연산자, == 에 대한 동작을 정의합니다.
        if isinstance(other, Variable):
            other = other.data
        return self.data == other

    def __ne__(self, other): # 부등호 연산자, != 에 대한 동작을 정의합니다.
        if isinstance(other, Variable):
            other = other.data
        return self.data != other
    
    def __lt__(self, other): # 보다 작음 연산자, < 에 대한 동작을 정의합니다.
        if isinstance(other, Variable):
            other = other.data
        return self.data < other
    
    def __gt__(self, other): # 보다 큼 연산자, > 에 대한 동작을 정의합니다.
        if isinstance(other, Variable):
            other = other.data
        return self.data > other
    
    def __le__(self, other): # 보다 작거나 같음 연산자, <= 에 대한 동작을 정의합니다.
        if isinstance(other, Variable):
            other = other.data
        return self.data <= other
    
    def __ge__(self, other): # 크거나 같음 연산자, >= 에 대한 동작을 정의합니다.
        if isinstance(other, Variable):
            other = other.data
        return self.data >= other

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return dezero.functions.transpose(self, axes)

    def sum(self, axis=None, keepdims=False):
        return dezero.functions.sum(self, axis, keepdims)

    def to_cpu(self):
        if self.data is not None:
            self.data = dezero.cuda.as_numpy(self.data)

    def to_gpu(self):
        if self.data is not None:
            self.data = dezero.cuda.as_cupy(self.data)

    
class Parameter(Variable):
    pass


def as_array(x, array_module=np):
    if np.isscalar(x):
        return array_module.array(x)
    return x


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    else:
        return Variable(obj)


class Function():
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]

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
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


# =============================================================================
# 사칙연산 / 연산자 오버로드
# =============================================================================


class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return (y,)

    def backward(self, gy):
        x0, x1 = self.inputs
        return x1 * gy, x0 * gy


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


class Sub(Function):
    def forward(self, x0, x1):
        return x0 - x1

    def backward(self, gy):
        return gy, -gy


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1


class Pow(Function):
    def __init__(self, ndim):
        super().__init__()
        self.ndim = ndim

    def forward(self, x):
        y = x ** self.ndim
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = self.ndim * x**(self.ndim-1) * gy
        return gx


def add(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Add()(x0, x1)

def mul(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Mul()(x0, x1)

def neg(x):
    return Neg()(x)

def sub(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Sub()(x1, x0)

def div(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Div()(x1, x0)

def pow(x, ndim):
    return Pow(ndim)(x)


def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
    
    Variable.__getitem__ = dezero.functions.get_item