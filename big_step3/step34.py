import numpy as np
import matplotlib.pyplot as plt
from parent import print
from dezero import Variable
import dezero.functions as F


x = Variable(np.linspace(-7, 7, 200))
y = F.sin(x)
y.backward(create_graph=True)

logs = [y.data]

for i in range(3):
    logs.append(x.grad.data)
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)


# 그래프 그리기
labels = ["y=sin(x)", "y'", "y''", "y'''"]
colors = [ 
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    '#bcbd22', '#17becf'
] # dflt_cycle # ad[0].get_color() 

for i, v in enumerate(logs):
    ad = plt.plot(x.data, v, label=labels[i])
    plt.text(x.data[0], v[0], labels[i], fontsize=20, color=colors[i])

plt.legend(loc='lower right')
plt.get_current_fig_manager().window.showMaximized() # pip install pyqt5
plt.show()