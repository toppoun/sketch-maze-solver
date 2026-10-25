from constants import *
from collections import deque

def remove_outside(binary):
    result = binary.copy()
    h, w = binary.shape
    que = deque()
    visited = [[False for _ in range (w)] for _ in range(h)]
    for x in range(w):
        if binary[0][x] == 1:
            que.append((0, x))
            visited[0][x] = True
            result[0][x] = 0
        if binary[h-1][x] == 1:
            que.append((h-1, x))
            visited[h-1][x] = True
            result[h-1][x] = 0
    for y in range(h):
        if binary[y][0]:
            que.append((y, 0))
            visited[y][0] = True
            result[y][0] = 0
        if binary[y][w-1]:
            que.append((y, w-1))
            visited[y][w-1] = True
            result[y][w-1] = 0
    while que:
        i, j = que.pop()
        for di, dj in DIR4:
            ni = i + di
            nj = j + dj
            if not(0 <= ni < h and 0 <= nj < w):
                continue
            if visited[ni][nj]:
                continue
            if binary[ni][nj] == 0:
                continue
            que.append((ni, nj))
            visited[ni][nj] = True
            result[ni][nj] = 0
    return result