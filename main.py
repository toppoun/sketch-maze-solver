from pathlib import Path
import cv2
import numpy as np

def read_maze(image_path, output_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)

    # グレースケール
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二値化
    _, thresh = cv2.threshold(
        gray, 127, 255, cv2.THRESH_BINARY
    )
    print(thresh)

    # 保存
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{image_path.stem}_binary.png"
    cv2.imwrite(str(output_file), thresh)

def main():
    input_path = Path("data/input/maze1.png")
    output_path = Path("data/output")
    read_maze(input_path, output_path)

if __name__ == "__main__":
    main()