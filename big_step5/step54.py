import numpy as np
from parent import print
from dezero import test_mode
import dezero.functions as F

# 드롭아웃과 테스트 모드

x = np.ones(5)
print(x)

y = F.dropout(x)
print(y)

with test_mode():
    y = F.dropout(x)
    print(y)