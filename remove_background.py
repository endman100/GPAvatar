import sys
import os
sys.path.append("./DiffMatte/")

import cv2
import numpy as np
from ultralytics import YOLO



# 定義去背函數
def remove_background(video_path, background_image_path, output_path):
    # 讀取影片
    cap = cv2.VideoCapture(video_path)
    
    # 讀取背景圖片並調整大小
    background = cv2.imread(background_image_path)
    h, w = background.shape[:2]
    ret, frame = cap.read()
    if not ret:
        print("無法讀取影片")
        return
    
    # 設定影片編碼器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 編碼器
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))

    # 設定影片mask編碼器
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 編碼器
    # mask_out = cv2.VideoWriter(output_path.replace(".mp4", "_mask.mp4"), fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))

    index = 0
    while cap.isOpened():
        print(index, end="\r")
        ret, frame = cap.read()        
        if not ret:
            break
        frame = cv2.resize(frame, (w, h))
        
                
        # 使用遮罩混和背景與前景
        mask = get_mix_mask(frame, index)
        mask = apply_linear_gradient(mask)

        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        _mask = mask.astype(np.float32) / 255
        mix_frame = frame.astype(np.float32) * (_mask) + background.astype(np.float32) * (1 - _mask)
        mix_frame = mix_frame.astype(np.uint8)

        cv2.imshow("mix_frame", cv2.resize(mix_frame, (0, 0), fx=0.5, fy=0.5))
        cv2.waitKey(1)

        
        # 寫入新影格到影片
        out.write(mix_frame)
        index += 1

    cap.release()
    out.release()
    # mask_out.release()


def apply_linear_gradient(mask):
    eroded_mask = cv2.erode(mask, np.ones((10, 10), np.uint8), iterations=1)

    # 找出侵蝕前後的邊界
    boundary_before = cv2.Canny(mask, 100, 200)
    boundary_after = cv2.Canny(eroded_mask, 100, 200)
    
    # 計算每個像素到侵蝕前邊界的距離
    dist_to_before = cv2.distanceTransform(255 - boundary_before, cv2.DIST_L2, 5)
    # 計算每個像素到侵蝕後邊界的距離
    dist_to_after = cv2.distanceTransform(255 - boundary_after, cv2.DIST_L2, 5)
    
    # 正規化距離到 0-1 範圍
    dist_to_before = cv2.normalize(dist_to_before, None, 0, 1.0, cv2.NORM_MINMAX)
    dist_to_after = cv2.normalize(dist_to_after, None, 0, 1.0, cv2.NORM_MINMAX)
    
    # 計算線性過渡，避免除以零
    sum_distances = dist_to_before + dist_to_after
    with np.errstate(divide='ignore', invalid='ignore'):
        gradient = np.divide(dist_to_before, sum_distances, out=np.zeros_like(dist_to_before), where=sum_distances!=0)
    
    gradient = (gradient * 255).astype(np.uint8)
    
    # 只在差異區域應用漸變
    difference = cv2.absdiff(mask, eroded_mask)
    result = np.zeros_like(mask, dtype=np.uint8)
    result[difference > 0] = gradient[difference > 0]
    result[difference == 0] = mask[difference == 0]
    return result

from PIL import Image
from re import findall
from os.path import join as opj
from torchvision.transforms import functional as F
from detectron2.engine import default_argument_parser
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
import torch
from ultralytics import YOLO
def get_mix_mask(frame, index):
    mask_a = get_mask_binary(frame)
    mask_b = get_mask_from_file(index)
    mask_b = cv2.erode(mask_b, np.ones((20, 20), np.uint8), iterations=1)
    mask = cv2.bitwise_or(mask_a, mask_b)
    return mask


def infer_one_image(model, input, save_dir=None):
    """
    Infer the alpha matte of one image.
    Input:
        model: the trained model
        image: the input image
        trimap: the input trimap
    """
    output = model(input)
    output = output.astype(np.uint8)
    return output

def init_model(model, checkpoint, device, sample_strategy):
    """
    Initialize the model.
    Input:
        config: the config file of the model
        checkpoint: the checkpoint of the model
    """
    cfg = LazyConfig.load(model)
    if sample_strategy is not None:
        cfg.difmatte.args["use_ddim"] = True if "ddim" in sample_strategy else False
        cfg.diffusion.steps = int(findall(r"\d+", sample_strategy)[0])
    
    model = instantiate(cfg.model)
    diffusion = instantiate(cfg.diffusion)
    cfg.difmatte.model = model
    cfg.difmatte.diffusion = diffusion
    difmatte = instantiate(cfg.difmatte)
    difmatte.to(device)
    difmatte.eval()
    DetectionCheckpointer(difmatte).load(checkpoint)
    
    return difmatte

def get_data(image, trimap):
    # image = Image.open(image_dir).convert('RGB')
    # image = F.to_tensor(image).unsqueeze(0)
    # trimap = Image.open(trimap_dir).convert('L')
    # trimap = F.to_tensor(trimap).unsqueeze(0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = F.to_tensor(image).unsqueeze(0)
    trimap = F.to_tensor(trimap).unsqueeze(0)

    return {
        'image': image,
        'trimap': trimap
    }

def get_mask_from_file(index, shape=(1280, 1280)):
    mask_dir = "./demos/mask/test1_test1/"
    mask_path = os.path.join(mask_dir, f"{index}.png")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, shape)
    return mask
# seg_model = YOLO("person_yolov8m-seg.pt")
# matting_model = init_model("./DiffMatte/configs/ViTS_1024.py", "DiffMatte_ViTS_Com_1024.pth", "cuda", "ddim2")
def create_trimap(alpha_mask, erosion_iter=10, dilation_iter=10):
    # alpha_mask: 需要傳入的二值化 alpha mask
    # erosion_iter: 腐蝕操作的次數
    # dilation_iter: 膨脹操作的次數

    # 將 alpha mask 二值化
    _, binary_mask = cv2.threshold(alpha_mask, 127, 255, cv2.THRESH_BINARY)

    # 創建 trimap，初始化為全 255（表示未知區域）
    trimap = np.full(binary_mask.shape, 0, np.uint8)

    

    # 膨脹操作以獲得確定的背景區域（確定區域）
    bg = cv2.dilate(binary_mask, None, iterations=dilation_iter)
    trimap[bg > 0] = 128  # 背景設置為 0

    # 腐蝕操作以獲得確定的前景區域（確定區域）
    fg = cv2.erode(binary_mask, None, iterations=erosion_iter)
    trimap[fg == 255] = 255  # 前景設置為 255



    return trimap
def get_mask(frame):
    seg_mask = get_segmentation(frame)
    black_mask = cv2.inRange(frame, (0, 0, 0), (0, 0, 0))
    seg_mask[black_mask == 255] = 0
    trimap = create_trimap(seg_mask)
    input = get_data(frame, trimap)
    output = infer_one_image(matting_model, input)
    output[black_mask == 255] = 0
    
    # cv2.imshow("seg_mask", cv2.resize(seg_mask, (0, 0), fx=0.5, fy=0.5))
    # cv2.imshow("trimap", cv2.resize(trimap, (0, 0), fx=0.5, fy=0.5))
    # cv2.imshow("output", cv2.resize(output, (0, 0), fx=0.5, fy=0.5))
    # cv2.imshow("black_mask", cv2.resize(black_mask, (0, 0), fx=0.5, fy=0.5))
    # cv2.waitKey(1)
    return output, seg_mask, trimap
def get_mask_binary(frame):    
    black_mask = cv2.inRange(frame, (0, 0, 0), (20, 20, 20))
    black_mask = 255 - black_mask

    black_mask[-1, :] = 255
    black_mask = cv2.copyMakeBorder(black_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)

    filled_image = np.zeros_like(black_mask)
    cv2.drawContours(filled_image, [max_contour], -1, 255, thickness=cv2.FILLED)    
    # cv2.imshow("black_mask", cv2.resize(black_mask, (0, 0), fx=0.5, fy=0.5))
    # cv2.imshow("filled_image", cv2.resize(filled_image, (0, 0), fx=0.5, fy=0.5))
    # cv2.waitKey(1)
    return filled_image[1:-1, 1:-1]
# def insert_points(contour, max_distance=10):
#     new_contour = []
#     for i in range(len(contour)):
#         new_contour.append(contour[i])
#         next_i = (i + 1) % len(contour)
#         dist = np.linalg.norm(contour[next_i][0] - contour[i][0])
#         while dist > max_distance:
#             # 計算中間點
#             mid_point = (contour[i][0] + contour[next_i][0]) / 2
#             new_contour.append([mid_point.astype(np.int32)])
#             dist = np.linalg.norm(mid_point - contour[i][0])
#             contour[i][0] = mid_point
#     return np.array(new_contour, dtype=np.int32)
def angle_between(p1, p2, p3):
    """
    計算由三個點形成的夾角，p2 為頂點
    :param p1: 第一個點 (x1, y1)
    :param p2: 頂點 (x2, y2)
    :param p3: 第三個點 (x3, y3)
    :return: 夾角（以度為單位）
    """
    # 計算向量
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

    # 計算向量的內積和模長
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # 檢查向量的模長是否為零
    if norm_v1 == 0 or norm_v2 == 0:
        raise ValueError(f"向量的模長不能為零，請確保點不重合。 p1: {p1}, p2: {p2}, p3: {p3}")

    # 計算夾角的餘弦值
    cos_theta = dot_product / (norm_v1 * norm_v2)
    
    # 防止因浮點數精度問題導致cos_theta超出[-1, 1]範圍
    cos_theta = np.clip(cos_theta, -1, 1)

    # 計算夾角（以弧度為單位），再轉換為度
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)

    #離鈍角近的角度
    angle_deg = min(angle_deg, abs(180 - angle_deg))
    return angle_deg

def angle_based_smoothing(contour, search_points, distance_threshold, distance_search_points_ratio=0.5):
    new_contour = [contour[0]]
    index = 1 # 從第二個點開始
    while(index < len(contour)):
        preview_point = new_contour[-1][0]
        current_point = contour[index][0]
        min_angle_change = float('inf')
        best_point = current_point
        best_index = index
        
        #搜索目標點數
        for j in range(index + 1, index + search_points + 1):
            if j >= len(contour):
                break
            candidate_point = contour[j][0]
            angle = angle_between(preview_point, current_point, candidate_point)
            if angle < min_angle_change:
                min_angle_change = angle
                best_point = candidate_point
                best_index = j
        
        #搜索所有後的點 直到距離超過閾值
        distance_search_points = int(distance_search_points_ratio * len(contour))
        for j in range(index + 1, distance_search_points):
            if j >= len(contour):
                break
            candidate_point = contour[j][0]
            distance = np.linalg.norm(candidate_point - current_point)
            if distance <= distance_threshold:
                angle = angle_between(preview_point, current_point, candidate_point)
                if angle < min_angle_change:
                    min_angle_change = angle
                    best_point = candidate_point
                    best_index = j
        
        new_contour.append([best_point])
        index = best_index + 1
        
    return np.array(new_contour, dtype=np.int32)

def get_segmentation(frame):
    # 裁切區域內 人體區域外的部分用背景圖片填充
    h, w = frame.shape[:2]
    result = seg_model.predict(frame)[0]  # predict on an image
    masks = result.masks

    # find the largest mask
    max_mask, max_mask_area = None, 0
    for mask in masks:
        mask_area = torch.sum(mask.data)
        if  mask_area > max_mask_area:
            max_mask_area = mask_area
            max_mask = mask.data

    max_mask_np = max_mask.cpu().numpy() * 255
    max_mask_np = np.transpose(max_mask_np, (1, 2, 0)).astype(np.uint8)
    max_mask_np = cv2.resize(max_mask_np, (w, h))
    return max_mask_np


import argparse
import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=r".\results\OneModel\test1_test1.mp4", help='Input video path.')
    parser.add_argument('--output_path', type=str, default=r".\results\OneModel\test1_test1_bg.mp4", help='Output video path.')
    parser.add_argument('--background_path', type=str, default=r".\demos\backgrounds\photo_2024-06-24_05-49-29_inpaint.png", help='Background image path.')
    args = parser.parse_args()
    remove_background(args.input_path, args.background_path, args.output_path)


