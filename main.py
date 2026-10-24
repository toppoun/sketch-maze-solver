from pathlib import Path
import cv2
import numpy as np
from collections import deque

DIR4 = [(1,0),(-1,0),(0,1),(0,-1)]
DIR8 = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
state = {"start": None, "goal": None}



def read_maze(image_path, output_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)

    # グレースケール
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二値化
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)


    output_path.mkdir(parents=True, exist_ok=True)

    # 二値画像
    cv2.imwrite(
        str(output_path / f"{image_path.stem}_binary.png"),
        thresh
    )

    # 通路を1に変換
    binary = (thresh > 0).astype(np.uint8)

    # 外側を黒くする
    binary = remove_outside(binary)
    cv2.imwrite(
    str(output_path / f"{image_path.stem}_masked.png"),
    binary * 255
    )

    road = Zhang_Suen(binary)

    road_img = (road * 255).astype(np.uint8)
    cv2.imwrite(
        str(output_path / f"{image_path.stem}_zhangsuen.png"),
        road_img
    )

    display = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    # 中心線の部分だけ灰色にする
    # display[road == 1] = (160, 160, 160)
    cv2.imwrite(
        str(output_path / f"{image_path.stem}_display.png"),
        display
    )

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 800, 800)
    cv2.imshow("Image", display)
    cv2.setMouseCallback('Image', on_mouse, (road, binary, display))
    cv2.waitKey(0) 
    cv2.destroyAllWindows()


def neighbours(x, y, image):
    return [image[x-1][y], image[x-1][y+1], image[x][y+1], image[x+1][y+1],
             image[x+1][y], image[x+1][y-1], image[x][y-1], image[x-1][y-1]]

def count_zero_to_one(points):
    res = 0
    for i in range(8):
        if points[i] == 0 and points[(i + 1) % 8] == 1:
            res += 1
    return res

def Zhang_Suen(binary_imag):
    imag = binary_imag.copy()
    h, w = imag.shape
    changed1 = changed2 = 1
    while changed1 or changed2:
        changed1 = 0
        changed = []
        for x in range(1, h - 1):
            for y in range(1, w - 1):
                p2, p3, p4, p5, p6, p7, p8, p9 = points = neighbours(x, y, imag)
                if(imag[x][y] == 1 and 2 <= sum(points) <= 6 and count_zero_to_one(points) == 1 and p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0):
                    changed.append((x, y))
                    changed1 = 1

        for x, y in changed:
            imag[x][y] = 0

        changed = []
        changed2 = 0
        for x in range(1, h - 1):
            for y in range(1, w - 1):
                p2, p3, p4, p5, p6, p7, p8, p9 = points = neighbours(x, y, imag)
                if(imag[x][y] == 1 and 2 <= sum(points) <= 6 and count_zero_to_one(points) == 1 and p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0):
                    changed.append((x, y))
                    changed2 = 1
        for x, y in changed:
            imag[x][y] = 0
    return imag

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

def on_mouse(event, x, y, flags, param):
    road, binary, original = param

    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
        point = find_nearest_road(road, binary, x, y)

        if point is None:
            return

        display = original.copy()

        if event == cv2.EVENT_LBUTTONDOWN:
            state["start"] = point
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["goal"] = point

        if not (state["start"] is None or state["goal"] is None):
            path = solve_maze(road, state["start"], state["goal"])

            if path is not None:
                for x, y in path:
                    display[y, x] = (0, 255, 0)
            else:
                print("経路が見つかりません")

        if state["start"] is not None:
            cv2.circle(display, state["start"], 4, (0, 0, 255), -1)
        if state["goal"] is not None:
            cv2.circle(display, state["goal"], 4, (255, 0, 0), -1)

        cv2.imshow("Image", display)


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

def solve_maze(road, start, goal):
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
        




def main():
    input_path = Path("data/input/maze1.png")
    output_path = Path("data/output")
    read_maze(input_path, output_path)


if __name__ == "__main__":
    main()