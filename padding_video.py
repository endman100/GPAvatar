import cv2
import numpy as np

def resize_and_pad_video(input_path, output_path):
    # 打開影片
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 取得影片的訊息
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 編碼器
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 計算新的尺寸
    if width > height:
        new_width = 1024
        new_height = int(height * (1024 / width))
    else:
        new_height = 1024
        new_width = int(width * (1024 / height))

    # 計算填充邊界
    top = (1024 - new_height) // 2
    bottom = 1024 - new_height - top
    left = (1024 - new_width) // 2
    right = 1024 - new_width - left

    # 設定輸出影片
    out = cv2.VideoWriter(output_path, fourcc, fps, (1024, 1024))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 調整尺寸
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # 添加白色邊框
        padded_frame = cv2.copyMakeBorder(
            resized_frame, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )

        # 寫入新的影片
        out.write(padded_frame)

    # 釋放資源
    cap.release()
    out.release()
    cv2.destroyAllWindows()


import argparse
import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=r".\demos\drivers\test2\c46fdcf5-558b-422d-b4f5-d15f97701d60.mp4", help='Input video path.')
    parser.add_argument('--output', type=str, default=r".\demos\drivers\test2\test2.mp4", help='Output video path.')
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(args.input):
        print('Error: Input video not found.')
    else:
        resize_and_pad_video(args.input, args.output)