from pathlib import Path
import cv2
import numpy as np
from algorithms import zhang_suen as algorithm
from preprocess import remove_outside

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

    prepared = algorithm.prepare(binary)

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
    cv2.setMouseCallback('Image', on_mouse, (prepared, display))
    cv2.waitKey(0) 
    cv2.destroyAllWindows()



def on_mouse(event, x, y, flags, param):

    prepared, original = param

    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
        point = algorithm.select_point(prepared, (x, y))

        if point is None:
            return

        display = original.copy()

        if event == cv2.EVENT_LBUTTONDOWN:
            state["start"] = point
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["goal"] = point

        if not (state["start"] is None or state["goal"] is None):
            path = algorithm.solve(prepared, state["start"], state["goal"])

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


def main():
    # input_path = Path("data/input/maze1.png")
    input_path = Path("data/external/image.png")
    output_path = Path("data/output")
    read_maze(input_path, output_path)


if __name__ == "__main__":
    main()