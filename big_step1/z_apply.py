from step5to9 import np, Variable, Function, square
import matplotlib.pyplot as plt


class Cubic(Function):
    def forward(self, x):
        y = x ** 3
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 3 * x**2 * gy
        return gx
        

class Quadruple(Function):
    def forward(self, x):
        y = x ** 4
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 4 * x**3 * gy
        return gx


class Square_n(Function):
    def __init__(self, ndim):
        super().__init__()
        self.ndim = ndim

    def forward(self, x):
        y = x ** self.ndim
        return y

    def backward(self, gy):
        x = self.input.data
        gx = self.ndim * x**(self.ndim-1) * gy
        return gx


def cubic(x):
    return Cubic()(x)

def quadruple(x):
    return Quadruple()(x)

def square_n(x, ndim):
    return Square_n(ndim)(x)

s_range = 1, 5
x0 = np.arange(*s_range, dtype=np.float32)
x = Variable(x0.reshape(1, -1))
# y_2s = square(x)
# y = cubic(y_2s)
# y.backward()
# print(y_2s.grad)
# print(x.grad)

output, grads = None, None

for n in x0.tolist():
    y = square_n(x, n)
    y.backward()
    
    if output is None:
        output = y.data
        grads = x.grad
    else:
        output = np.concatenate((output, y.data), axis=0)
        grads = np.concatenate((grads, x.grad), axis=0)


# output = output.T
# grads = grads.T
print(output)
print(grads)


fig, axes = plt.subplots(1, 2)
t = np.arange(*s_range)

for i, (result, y_label) in enumerate(zip([output, grads], ['output', 'grads'])):
    for idx, n in enumerate(t.tolist()):
        axes[i].plot(t, result[idx], label=f'{n}^2')
        for x_tick in range(len(t)):
           height = result[idx][x_tick]
           axes[i].text(t[x_tick], height + 0.25, '%d' %height, ha='center', va='bottom', size = 12)

    axes[i].set_xlim(s_range[0], s_range[1]-1)
    axes[i].set_xticks(list(range(1, 5)))
    axes[i].set_xlabel('n_square')
    axes[i].set_ylabel(y_label)
    axes[i].grid(True)
    axes[i].legend(loc='upper left')

# plt.tight_layout()
plt.show()