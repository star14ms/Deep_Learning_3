from parent import print
from dezero.utils import get_conv_outsize, pair

# CNN 메커니즘 (1)

H, W = 4, 4 # 입력 형상
KH, KW = 3, 3 # 커널 형상
SH, SW = 1, 1 # 스트라이드(세로 방향과 스트라이드와 가로 방향 스트라이드)
PH, PW = 1, 1 # 패딩 (세로 방향 패딩과 가로 방향 패딩)

OH = get_conv_outsize(H, KH, SH, PH)
OW = get_conv_outsize(W, KW, SW, PW)

print(OH, OW)


print(pair(1))
print(pair((1, 2)))