from constants import DIR4, DIR8
from numba import njit
from collections import deque

def prepare(binary):
    road = Zhang_Suen(binary)
    return binary, road

def solve(prepared, start, goal):
    _, road = prepared
    h, w = road.shape
    parent = {start: None}
    que = deque([start])
    while que:
        j, i = que.popleft()
        if (j, i) == goal:
            break
        for di, dj in DIR8:
            ni = i + di
            nj = j + dj
            if not (0 <= ni < h and 0 <= nj < w):
                continue
            if (nj, ni) in parent:
                continue
            if road[ni][nj] == 0:
                continue

            parent[(nj, ni)] = (j, i)
            que.append((nj, ni))

    if goal not in parent:
        return None

    path = []
    cur = goal

    while cur is not None:
        path.append(cur)
        cur = parent[cur]

    path.reverse()

    return path


@njit
def neighbours(x, y, image):
    return [image[x-1][y], image[x-1][y+1], image[x][y+1], image[x+1][y+1],
             image[x+1][y], image[x+1][y-1], image[x][y-1], image[x-1][y-1]]

@njit
def count_zero_to_one(points):
    res = 0
    for i in range(8):
        if points[i] == 0 and points[(i + 1) % 8] == 1:
            res += 1
    return res

@njit
def Zhang_Suen(binary):
    road = binary.copy()
    h, w = road.shape
    changed1 = changed2 = 1

    # start = perf_counter()
    iterations = 0

    while changed1 or changed2:

        iterations += 1
        # if iterations % 10 == 0:
        # elapsed = perf_counter() - start
        # print(f"{iterations=}, {elapsed=:.2f}s", flush=True)
        print("iterations:", iterations)
        
        changed1 = 0
        changed = []
        for x in range(1, h - 1):
            for y in range(1, w - 1):
                if road[x, y] == 0:
                    continue
                p2, p3, p4, p5, p6, p7, p8, p9 = points = neighbours(x, y, road)
                if(road[x][y] == 1 and 2 <= sum(points) <= 6 and count_zero_to_one(points) == 1 and p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0):
                    changed.append((x, y))
                    changed1 = 1

        for x, y in changed:
            road[x][y] = 0

        changed = []
        changed2 = 0
        for x in range(1, h - 1):
            for y in range(1, w - 1):
                if road[x, y] == 0:
                    continue
                p2, p3, p4, p5, p6, p7, p8, p9 = points = neighbours(x, y, road)
                if(road[x][y] == 1 and 2 <= sum(points) <= 6 and count_zero_to_one(points) == 1 and p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0):
                    changed.append((x, y))
                    changed2 = 1
        for x, y in changed:
            road[x][y] = 0
    return road


def select_point(prepared, point):
    binary, road = prepared
    x, y = point
    return find_nearest_road(road, binary, x, y)

def find_nearest_road(road, binary, x, y):
    h, w = road.shape

    if not (0 <= x < w and 0 <= y < h):
        return None

    if binary[y][x] == 0:
        return None

    que = deque([(x, y)])
    visited = [[False] * w for _ in range(h)]
    visited[y][x] = True

    while que:
        j, i = que.popleft()

        if road[i][j] == 1:
            return j, i

        for di, dj in DIR4:
            ni = i + di
            nj = j + dj

            if not (0 <= ni < h and 0 <= nj < w):
                continue
            if visited[ni][nj]:
                continue
            if binary[ni][nj] == 0:
                continue

            que.append((nj, ni))
            visited[ni][nj] = True

    return None
