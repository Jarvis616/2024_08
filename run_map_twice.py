import argparse
import time
from pathlib import Path

import cv2
import torch
import math
import os
import threading
import pygame
import mediapipe as mp
import torch.backends.cudnn as cudnn
from torch import nn
from numpy import random
import pyrealsense2 as rs
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box, my_plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.datasets import letterbox
from collections import deque, Counter
from PIL import ImageFont, ImageDraw, Image
from move_lite3 import HeartBeat, ControlRobot
import rospy
import joblib
from playsound import playsound

from angle_cx import align_with_white_line, move_ball_to_position
import time
from rostopy import publisher_init, publish_cmd_vel
from torchvision import transforms

font_path = './YZGGPB.ttf'
font = ImageFont.truetype(font_path,50)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 8)  # 8个类别

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 16*4*4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义PyTorch模型
class GestureRecognitionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GestureRecognitionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# 加载训练好的模型
input_dim = 63  # 21个关键点，每个关键点有x, y, z三个坐标
output_dim = 7  # 手势类别数量
hand_model = GestureRecognitionModel(input_dim, output_dim)
hand_model.load_state_dict(torch.load('best_gesture_recognition_model_7_4_01.pth'))
hand_model.eval()

# 加载标准化模型
scaler = joblib.load('scaler_7_4_01.pkl')

# 处理新图片
def process_image(image, img0, xx, yy):

    if image is None:
        return None
    
    # 转换图像颜色
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 处理图像以检测手部关键点
    results = hands.process(image_rgb)
    
    # 如果检测到手部关键点
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]  # 假设只有一个检测到的手
        
        # 准备数据进行预测
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z if landmark.z is not None else 0])
        
        # 在原图上绘制手部关键点
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * image.shape[1]) + xx
            y = int(landmark.y * image.shape[0]) + yy 
            cv2.circle(img0, (x, y), 2, (0, 255, 0), -1)

        # 数据预处理：标准化输入数据
        landmarks = scaler.transform(np.array([landmarks]))
        # 将数据转换为PyTorch张量
        landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32)

         # 使用加载的模型进行预测
        with torch.no_grad():
            prediction = hand_model(landmarks_tensor)
            predicted_label = torch.argmax(prediction, dim=1).item()

        
        return predicted_label
    else:
        return None

def xyxy_in_box(xyxy_small, xyxy_large):
    
    c1, c2 = (int(xyxy_small[0].item()), int(xyxy_small[1].item())), (int(xyxy_small[2].item()), int(xyxy_small[3].item()))

    in_x1 = max(c1[0],c2[0])
    in_x2 = min(c1[0],c2[0])
    in_y1 = max(c1[1],c2[1])
    in_y2 = min(c1[1],c2[1])

    c3, c4 = (int(xyxy_large[0].item()), int(xyxy_large[1].item())), (int(xyxy_large[2].item()), int(xyxy_large[3].item()))
    out_x1 = max(c3[0],c4[0])
    out_x2 = min(c3[0],c4[0])
    out_y1 = max(c3[1],c4[1])
    out_y2 = min(c3[1],c4[1])

    return in_x1 < out_x1 and in_x2 > out_x2 and in_y1 < out_y1 and in_y2 > out_y2

def calculate_other_two_corners(x1, y1, x2, y2):
    width = abs(x2 - x1)
    height = abs(y2 - y1)

    if x1 <= x2 and y1 <= y2:
        # (x1, y1) 是左上角顶点
        x3, y3 = x2, y1
        x4, y4 = x1, y2
    elif x1 <= x2 and y1 >= y2:
        # (x1, y1) 是左下角顶点
        x3, y3 = x2, y2
        x4, y4 = x1, y1
    elif x1 >= x2 and y1 <= y2:
        # (x1, y1) 是右上角顶点
        x3, y3 = x2, y2
        x4, y4 = x1, y1
    elif x1 >= x2 and y1 >= y2:
        # (x1, y1) 是右下角顶点
        x3, y3 = x2, y1
        x4, y4 = x1, y2

    return (x3, y3), (x4, y4)

def caculate_angle(img0, xyxy_dashboard, xyxy_pointer, xyxy_sign):
    (x1, y1), (x2, y2) = (xyxy_dashboard[0].item(), xyxy_dashboard[1].item()), (xyxy_dashboard[2].item(), xyxy_dashboard[3].item())
    center_dashboard_x = (x1 + x2) / 2.0
    center_dashboard_y = (y1 + y2) / 2.0
    ratio = (y2 - y1) / (x2 - x1)
    #print("ratio", ratio)
    (x1, y1), (x2, y2) = (xyxy_sign[0].item(), xyxy_sign[1].item()), (xyxy_sign[2].item(), xyxy_sign[3].item())
    center_sign_x = (x1 + x2) / 2.0
    center_sign_y = (y1 + y2) / 2.0

    (x1, y1), (x2, y2) = (xyxy_pointer[0].item(), xyxy_pointer[1].item()), (xyxy_pointer[2].item(), xyxy_pointer[3].item())
    (x3, y3), (x4, y4) = calculate_other_two_corners(x1, y1, x2, y2)

    distance =[0, 0, 0, 0]
    distance[0] = math.sqrt((center_dashboard_x - x1)**2 + (center_dashboard_y - y1)**2)
    distance[1] = math.sqrt((center_dashboard_x - x2)**2 + (center_dashboard_y - y2)**2)
    distance[2] = math.sqrt((center_dashboard_x - x3)**2 + (center_dashboard_y - y3)**2)
    distance[3]= math.sqrt((center_dashboard_x - x4)**2 + (center_dashboard_y - y4)**2)
    min_index = distance.index(min(distance))
    if min_index == 0:
        (pointer_x1, pointer_y1) = (x1, y1)
        (pointer_x2, pointer_y2) = (x2, y2)
    elif min_index == 1:
        (pointer_x1, pointer_y1) = (x2, y2)
        (pointer_x2, pointer_y2) = (x1, y1)
    elif min_index == 2:
        (pointer_x1, pointer_y1) = (x3, y3)
        (pointer_x2, pointer_y2) = (x4, y4)
    elif min_index == 3:
        (pointer_x1, pointer_y1) = (x4, y4)
        (pointer_x2, pointer_y2) = (x3, y3)
    cv2.line(img0, (int(center_sign_x),int(center_sign_y)), (int(center_dashboard_x),int(center_dashboard_y)), (0, 255, 0))
    cv2.line(img0, (int(pointer_x1),int(pointer_y1)), (int(pointer_x2),int(pointer_y2)), (0, 0, 255))


    theta1 = np.arctan2(pointer_y2- pointer_y1, ratio*(pointer_x2 - pointer_x1))
    theta2 = np.arctan2(center_sign_y - center_dashboard_y, ratio*(center_sign_x - center_dashboard_x))

    theta = theta1 - theta2
    theta = np.degrees(theta)
    if theta < 0:
        theta = theta + 360
    #print(theta)
    if theta < 127.5:
        return 0
    elif theta >= 127.5 and theta < 230:
        return 1
    elif theta >= 230 and theta < 360:
        return 2

def move_to_dashboard_center(save_img,pub,v_x=0,v_y=0.2):
    distance = 10000
    dashboard_target_x = 200
    dashboard_target_y = 126

    
    # variable initialized here
    dashboard_classified_result = deque([-1] * 10,maxlen=10)

    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device2)['model']).to(device2).eval()

    # Get names and colors
    names = model2.module.names if hasattr(model2, 'module') else model2.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
    # Run inference
    if device2.type != 'cpu':
        model2(torch.zeros(1, 3, imgsz2, imgsz2).to(device2).type_as(next(model2.parameters())))  # run once
    old_img_w = old_img_h = imgsz2
    old_img_b = 1

    t0 = time.time()  # Initialize t0

    try:
        while True:
            if rospy.is_shutdown():
                break
            # 如果距离足够小，认为到达目标点，停止运动
            if abs(distance) < 10:
                publish_cmd_vel(pub, -1, 0, 0, 0)
                break
            
            # Wait for a new frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert frame to numpy array
            
            img0 = np.asanyarray(color_frame.get_data())

            # img0 = cv2.imread("gesture/gesture04.jpg")
            img = letterbox(img0, 640, 32)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            #img = img0.transpose(2, 0, 1)  # HWC to CHW
            img = torch.from_numpy(img).to(device2)
            img = img.half() if half2 else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            #print(img.ndimension())
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device2.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model2(img, augment=opt2.augment)[0]



            # Inference
            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                pred = model2(img, augment=opt2.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt2.conf_thres, opt2.iou_thres, classes=opt2.classes, agnostic=opt2.agnostic_nms)

            # Process detections
            move_x = None
            move_y = None
            for i, det in enumerate(pred):  # detections per image

                if len(det):
                    # Separate detections by class
                    det_dict = {0: [], 1: [], 2: []}
                    for *xyxy, conf, cls in det:
                        det_dict[int(cls)].append((*xyxy, conf))

                    # Sort detections by confidence for each class
                    for cls in det_dict.keys():
                        det_dict[cls] = sorted(det_dict[cls], key=lambda x: x[-1], reverse=True)

                    # Keep highest confidence detection for label=0
                    best_label0 = None
                    xyxy_sign = None
                    xyxy_pointer = None
                    if len(det_dict[0]) > 0:
                        best_label0 = max(det_dict[0], key=lambda x: x[-1])
                        best_xyxy, best_conf = best_label0[:4], best_label0[-1]
                        if best_conf > 0.6:
                            label = f'{names[int(0)]} {best_conf:.2f}' 
                            plot_one_box(best_xyxy, img0, label=label, color=colors[int(0)], line_thickness=1)
                            (x1, y1), (x2, y2) = (best_xyxy[0].item(), best_xyxy[1].item()), (best_xyxy[2].item(), best_xyxy[3].item())
                            center_dashboard_x = (x1 + x2) / 2.0
                            center_dashboard_y = (y1 + y2) / 2.0
                            print("dashboard_center",center_dashboard_x,center_dashboard_y)
                            distance_y = dashboard_target_x - center_dashboard_x
                            distance_x = dashboard_target_y - center_dashboard_y
                            #distance = math.sqrt(distance_x**2 + distance_y**2)
                            distance = distance_y
                            move_x, move_y = move_ball_to_position(0, distance_y)

                    
            # Show results
            cv2.imshow('RealSense', img0)
            cv2.waitKey(5)
            # 发布消息
            if move_x is not None and move_y is not None:
                publish_cmd_vel(pub, -1, 0, move_y, 0)
            else:
                publish_cmd_vel(pub, -1 , v_x, v_y, 0) 

    finally:
        print(f'Dashboard_Done. ({time.time() - t0:.3f}s)')
        cv2.destroyAllWindows()

def move_to_ball_center(save_img,pub):
    distance = 10000
    dashboard_target_x = 477
    dashboard_target_y = 377

    
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device5)['model']).to(device5).eval()

    # Get names and colors
    names = model5.module.names if hasattr(model5, 'module') else model5.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
    # Run inference
    if device5.type != 'cpu':
        model5(torch.zeros(1, 3, imgsz5, imgsz5).to(device5).type_as(next(model5.parameters())))  # run once
    old_img_w = old_img_h = imgsz5
    old_img_b = 1

    t0 = time.time()  # Initialize t0

    try:
        while True:
            # 如果距离足够小，认为到达目标点，停止运动
            if distance < 25:
                publish_cmd_vel(pub, -1, 0, 0, 0)
                break
            
            # Wait for a new frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert frame to numpy array
            
            img0 = np.asanyarray(color_frame.get_data())

            # img0 = cv2.imread("gesture/gesture04.jpg")
            img = letterbox(img0, 640, 32)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            #img = img0.transpose(2, 0, 1)  # HWC to CHW
            img = torch.from_numpy(img).to(device5)
            img = img.half() if half5 else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            #print(img.ndimension())
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device5.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model5(img, augment=opt5.augment)[0]



            # Inference
            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                pred = model5(img, augment=opt5.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt5.conf_thres, opt5.iou_thres, classes=opt5.classes, agnostic=opt5.agnostic_nms)

            # Process detections
            move_x = None
            move_y = None
            for i, det in enumerate(pred):  # detections per image

                if len(det):
                    # Separate detections by class
                    det_dict = {0: [], 1: [], 2: []}

                    for *xyxy, conf, cls in det:
                        det_dict[int(cls)].append((*xyxy, conf))

                    # Sort detections by confidence for each class
                    for cls in det_dict.keys():
                        det_dict[cls] = sorted(det_dict[cls], key=lambda x: x[-1], reverse=True)

                    # Keep highest confidence detection for label=0
                    best_label0 = None
                    xyxy_sign = None
                    xyxy_pointer = None
                    if len(det_dict[0]) > 0:
                        best_label0 = max(det_dict[0], key=lambda x: x[-1])
                        best_xyxy, best_conf = best_label0[:4], best_label0[-1]
                        if best_conf > 0.6:
                            label = f'{names[int(0)]} {best_conf:.2f}' 
                            plot_one_box(best_xyxy, img0, label=label, color=colors[int(0)], line_thickness=1)
                            (x1, y1), (x2, y2) = (best_xyxy[0].item(), best_xyxy[1].item()), (best_xyxy[2].item(), best_xyxy[3].item())
                            center_dashboard_x = (x1 + x2) / 2.0
                            center_dashboard_y = (y1 + y2) / 2.0
                            distance_y = dashboard_target_x - center_dashboard_x
                            distance_x = dashboard_target_y - center_dashboard_y
                            distance = math.sqrt(distance_x**2 + distance_y**2)
                            print("ball_center",center_dashboard_x,center_dashboard_y)
                            move_x, move_y = move_ball_to_position(distance_x, distance_y)

                    
            # Show results
            cv2.imshow('RealSense', img0)
            cv2.waitKey(5)
            # 发布消息
            if move_x is not None and move_y is not None:
                publish_cmd_vel(pub, -1, move_x, move_y, 0)
            else:
                publish_cmd_vel(pub, -1 , 0, 0.05, 0) 

    finally:
        print(f'Ball_Done. ({time.time() - t0:.3f}s)')
        cv2.destroyAllWindows()
        
def detect_gesture(save_img=False):
    # variable initialized here
    gesture_classified_result = deque([-1] * 14,maxlen=14)

    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()  # Initialize t0

    try:
        while True:
            # Wait for a new frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert frame to numpy array
            
            img0 = np.asanyarray(color_frame.get_data())

            # img0 = cv2.imread("gesture/gesture04.jpg")
            img = letterbox(img0, 640, 32)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            #img = img0.transpose(2, 0, 1)  # HWC to CHW
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            #print(img.ndimension())
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=opt1.augment)[0]


            # Inference
            t1 = time_synchronized()
            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=opt1.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, opt1.conf_thres, opt1.iou_thres, classes=opt1.classes, agnostic=opt1.agnostic_nms)
            t3 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    # Rescale boxes from img_size to img0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                     # Sort detections by confidence
                    det = det[det[:, 4].argsort(descending=True)]

                    # Keep only the highest confidence detection
                    det = det[:1]

                    # Only keep the highest confidence detection per class
                    seen_classes = {}
                    filtered_det = []
                    for *xyxy, conf, cls in det:
                        if cls.item() not in seen_classes:
                            seen_classes[cls.item()] = True
                            filtered_det.append((xyxy, conf, cls))

                    # Write results
                    for *xyxy, conf, cls in filtered_det:
                        if conf > 0.80:
                            #print("~~~~~~~~~~~~~~~~~",img0.shape)
                            small_x1 = int(xyxy[0][0].item() - (xyxy[0][2].item() - xyxy[0][0].item()) * 0.5)
                            small_x2 = int(xyxy[0][2].item() + (xyxy[0][2].item() - xyxy[0][0].item()) * 0.5)
                            small_y1 = int(xyxy[0][1].item() - (xyxy[0][3].item() - xyxy[0][1].item()) * 0.5)
                            small_y2 = int(xyxy[0][3].item() + (xyxy[0][3].item() - xyxy[0][1].item()) * 0.5)
                            small_x1 = max(small_x1, 0)
                            small_x2 = min(small_x2, 640)
                            small_y1 = max(small_y1, 0)
                            small_y2 = min(small_y2, 480)
                            #print(small_x1, small_y1, small_x2, small_y2)
                            #cv2.rectangle(img0,(small_x1,small_y1),(small_x2,small_y2),(0,255,0), thickness=1, lineType=cv2.LINE_AA)
                            img_hand = img0[small_y1:small_y2, small_x1:small_x2]
                            hand_result = process_image(img_hand, img0, small_x1, small_y1)
                            if hand_result is not None:
                                gesture_classified_result.append(int(hand_result))
                            label = f'{conf:.2f}'
                            # label = f'{names[int(cls)]} {conf:.2f}'
                            # print("++++++++",xyxy[0])
                            plot_one_box(xyxy[0], img0, label=label, color=colors[int(cls)], line_thickness=1)

            if len(gesture_classified_result) > 0:
                img_pil = Image.fromarray(cv2.cvtColor(img0,cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                count = Counter(gesture_classified_result) 
                most_common_num, most_common_count = count.most_common(1)[0] 
                if most_common_num == 0:
                    draw.text((150,50),"前进",font = font,fill = (255,0,0)) 
                elif most_common_num == 1:
                    draw.text((150,50),"后退",font = font,fill = (255,0,0)) 
                elif most_common_num == 2:
                    print("原地扭身")
                    draw.text((150,50),"原地扭身",font = font,fill = (255,0,0)) 
                elif most_common_num == 3 :
                    print("左平移")
                    draw.text((150,50),"左平移",font = font,fill = (255,0,0)) 
                elif most_common_num == 4 :
                    print("右平移")
                    draw.text((150,50),"右平移",font = font,fill = (255,0,0)) 
                elif most_common_num == 5 :
                    print("左旋转")
                    draw.text((150,50),"左旋转",font = font,fill = (255,0,0)) 
                elif most_common_num == 6 :
                    draw.text((150,50),"右旋转",font = font,fill = (255,0,0)) 
                img0 = cv2.cvtColor(np.array(img_pil),cv2.COLOR_RGB2BGR)


            # Show results
            cv2.imshow('RealSense', img0)
            cv2.waitKey(5)
            if most_common_num != -1:
                return most_common_num

            
    finally:
        print(f'Done. ({time.time() - t0:.3f}s)')

def detect_dashboard(save_img=False):
    most_common_num = -1
    
    # variable initialized here
    dashboard_classified_result = deque([-1] * 10,maxlen=10)

    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device2)['model']).to(device2).eval()

    # Get names and colors
    names = model2.module.names if hasattr(model2, 'module') else model2.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
    # Run inference
    if device2.type != 'cpu':
        model2(torch.zeros(1, 3, imgsz2, imgsz2).to(device2).type_as(next(model2.parameters())))  # run once
    old_img_w = old_img_h = imgsz2
    old_img_b = 1

    t0 = time.time()  # Initialize t0

    try:
        while True:
            # Wait for a new frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert frame to numpy array
            
            img0 = np.asanyarray(color_frame.get_data())
            #frame = None
            #ret , frame = cap.read()
            #if frame is None:
            #    continue
            #img0 = frame
            #img0 = img0[int(img0.shape[0] /2):, int(img0.shape[1] *1 / 3):int(img0.shape[1] *2 / 3)]
            # img0 = cv2.imread("gesture/gesture04.jpg")
            img = letterbox(img0, 640, 32)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            #img = img0.transpose(2, 0, 1)  # HWC to CHW
            img = torch.from_numpy(img).to(device2)
            img = img.half() if half2 else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            #print(img.ndimension())
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device2.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model2(img, augment=opt2.augment)[0]



            # Inference
            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                pred = model2(img, augment=opt2.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt2.conf_thres, opt2.iou_thres, classes=opt2.classes, agnostic=opt2.agnostic_nms)
            
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    # Separate detections by class
                    det_dict = {0: [], 1: [], 2: []}
                    for *xyxy, conf, cls in det:
                        det_dict[int(cls)].append((*xyxy, conf))

                    # Sort detections by confidence for each class
                    for cls in det_dict.keys():
                        det_dict[cls] = sorted(det_dict[cls], key=lambda x: x[-1], reverse=True)

                    # Keep top-3 highest confidence detections for label=0
                    best_label0_list = det_dict[0][:2] if len(det_dict[0]) >= 2 else det_dict[0]

                    # Create a list to store filtered detections for each label0
                    all_filtered_dets = []

                    for best_label0 in best_label0_list:
                        best_xyxy, best_conf = best_label0[:4], best_label0[-1]
                        if best_conf > 0.5:
                            label = f'{names[int(0)]} {best_conf:.2f}'
                            plot_one_box(best_xyxy, img0, label=label, color=colors[int(0)], line_thickness=1)

                        # Initialize a list to store the filtered detections for this label0
                        filtered_det = []

                        # Find the highest confidence label=1 within the current best_label0's box
                        if len(det_dict[1]) > 0:
                            for xyxy_conf in det_dict[1]:
                                xyxy, conf = xyxy_conf[:4], xyxy_conf[-1]
                                if xyxy_in_box(xyxy, best_xyxy):  # Function to check if xyxy is inside best_xyxy
                                    filtered_det.append((xyxy, conf, 1, best_xyxy))
                                    break  # Only need the highest confidence, so break after finding the first
                        
                        # Find the highest confidence label=2 within the current best_label0's box
                        if len(det_dict[2]) > 0:
                            for xyxy_conf in det_dict[2]:
                                xyxy, conf = xyxy_conf[:4], xyxy_conf[-1]
                                if xyxy_in_box(xyxy, best_xyxy):  # Function to check if xyxy is inside best_xyxy
                                    xyxy_sign = xyxy 
                                    filtered_det.append((xyxy, conf, 2, best_xyxy))
                                    break  # Only need the highest confidence, so break after finding the first

                        # Store the filtered detections for this label0 in all_filtered_dets
                        all_filtered_dets.append(filtered_det)

            
            
                    result = -1
                    left_dashboard_x1 = 10000
                    # Process the filtered detections
                    for filtered_det in all_filtered_dets:
                        xyxy_sign = None
                        xyxy_pointer = None
                        # Write results for label1 and label2 inside current label0 box
                        point_or_sign = 0
                        for *xyxy, conf, cls, best_xyxy in filtered_det:
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy[0], img0, label=label, color=colors[int(cls)], line_thickness=1)
                            if point_or_sign == 0:
                                xyxy_pointer = xyxy[0]
                                point_or_sign = point_or_sign + 1
                                xyxy_dashboard = best_xyxy
                            else:
                                xyxy_sign = xyxy[0]

                        # Calculate angle if both label1 and label2 are detected
                        if xyxy_sign is not None and xyxy_pointer is not None:
                            if int(xyxy_dashboard[0].item()) < left_dashboard_x1:
                                result = caculate_angle(img0, xyxy_dashboard, xyxy_pointer, xyxy_sign)
                                left_dashboard_x1 = xyxy_dashboard[0].item()
                    if result != -1:
                        dashboard_classified_result.append(result)

                    if len(dashboard_classified_result) > 0: 
                        img_pil = Image.fromarray(cv2.cvtColor(img0,cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(img_pil)     	
                        count = Counter(dashboard_classified_result) 
                        most_common_num, most_common_count = count.most_common(1)[0] 
                        
                        if most_common_num == 0:
                            print("仪表状态：偏低")
                            draw.text((150,50),"仪表状态：偏低",font = font,fill = (255,0,0))
                            #cv2.putText(img0, "status: low", (30, 50), 0, 1.5, (225, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                        elif most_common_num == 1:
                            print("仪表状态：正常")
                            draw.text((150,50),"仪表状态：正常",font = font,fill = (255,0,0))
                            #cv2.putText(img0, "status: mid", (30, 50), 0, 1.5, (225, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                        elif most_common_num == 2:
                            print("仪表状态：偏高")
                            draw.text((150,50),"仪表状态：偏高",font = font,fill = (255,0,0))

                            #cv2.putText(img0, "status: high", (30, 50), 0, 1.5, (225, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                        img0 = cv2.cvtColor(np.array(img_pil),cv2.COLOR_RGB2BGR)
            
            # Show results
            cv2.imshow('RealSense', img0)
            cv2.waitKey(5)
            if most_common_num != -1:
                return most_common_num
    finally:
        print(f'Dashboard_Done. ({time.time() - t0:.3f}s)')

def detect_number(save_img=False):
    most_common_num = -1
    robot.send_data(0x21010C02, 0, 0)
    time.sleep(0.1)
    robot.send_data(0x21010D05, 0, 0)
    time.sleep(0.1)
    # variable initialized here
    number_classified_result = deque([-1] * 10,maxlen=10)

    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device3)['model']).to(device3).eval()

    # Get names and colors
    names = model3.module.names if hasattr(model3, 'module') else model3.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
    # Run inference
    if device3.type != 'cpu':
        model3(torch.zeros(1, 3, imgsz3, imgsz3).to(device3).type_as(next(model3.parameters())))  # run once
    old_img_w = old_img_h = imgsz3
    old_img_b = 1

    t0 = time.time()  # Initialize t0
    # 加载模型并设置为评估模式
    lenet5_model = LeNet()
    lenet5_model.load_state_dict(torch.load('lenet_digit_classifier.pth'))
    lenet5_model.eval()

    # 预处理图像
    data_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    lenet5_model = lenet5_model.to(device)
    try:
        while True:
            robot.send_data(0x21010130, -20000, 0)
            time.sleep(0.05)        
            # Wait for a new frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert frame to numpy array
            
            img0 = np.asanyarray(color_frame.get_data())
            #img0 = img0[:, 0:int(img0.shape[1] *is_half_1 / is_half_2)]

            # img0 = cv2.imread("gesture/gesture04.jpg")
            img = letterbox(img0, 640, 32)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            #img = img0.transpose(2, 0, 1)  # HWC to CHW
            img = torch.from_numpy(img).to(device3)
            img = img.half() if half3 else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            #print(img.ndimension())
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device3.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model3(img, augment=opt3.augment)[0]



            # Inference
            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                pred = model3(img, augment=opt3.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt3.conf_thres, opt3.iou_thres, classes=opt3.classes, agnostic=opt3.agnostic_nms)
            

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    # Rescale boxes from img_size to img0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                     # Sort detections by confidence
                    det = det[det[:, 4].argsort(descending=True)]

                    # Keep only the highest confidence detection
                    det = det[:2]

                    # Only keep the highest confidence detection per class
                    filtered_det = []
                    for *xyxy, conf, cls in det:
                        filtered_det.append((xyxy, conf, cls))

                    result = -1
                    left_x = 10000
                    # Write results
                    for *xyxy, conf, cls in filtered_det:
                        if left_x > xyxy[0][0].item():
                            left_x = xyxy[0][0].item()
                            result = int(cls)
                            label = f'{conf:.2f}'
                            # label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy[0], img0, label=label, color=colors[int(cls)], line_thickness=1)
                            cropped_img = img0[int(xyxy[0][1]):int(xyxy[0][3]),int(xyxy[0][0]):int(xyxy[0][2])]
                            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                            resized_img = cv2.resize(cropped_img,(28,28))
                            resized_img = transforms.ToTensor()(resized_img).unsqueeze(0)
                            output = lenet5_model(resized_img.to(device))
                            _,predicted_label = torch.max(output,1)
                            result = predicted_label.item()
                    if result != -1:
                        number_classified_result.append(result)
            
            if len(number_classified_result) > 0:
                img_pil = Image.fromarray(cv2.cvtColor(img0,cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                count = Counter(number_classified_result) 
                most_common_num, most_common_count = count.most_common(1)[0] 
                if most_common_num == 0:
                    print("1号仪表盘")
                    draw.text((150,50),"1号仪表盘",font = font,fill = (255,0,0)) 
                elif most_common_num == 1:
                    print("2号仪表盘")
                    draw.text((150,50),"2号仪表盘",font = font,fill = (255,0,0)) 
                elif most_common_num == 2:
                    print("3号仪表盘2")
                    draw.text((150,50),"3号仪表盘",font = font,fill = (255,0,0)) 
                elif most_common_num == 3:
                    print("4号仪表盘")
                    draw.text((150,50),"4号仪表盘",font = font,fill = (255,0,0))
                elif most_common_num == 4:
                    print("5号仪表盘")
                    draw.text((150,50),"5号仪表盘",font = font,fill = (255,0,0)) 
                elif most_common_num == 5:
                    print("6号仪表盘")
                    draw.text((150,50),"6号仪表盘",font = font,fill = (255,0,0)) 
                elif most_common_num == 6:
                    print("7号仪表盘")
                    draw.text((150,50),"7号仪表盘",font = font,fill = (255,0,0))
                elif most_common_num == 7:
                    print("8号仪表盘")
                    draw.text((150,50),"8号仪表盘",font = font,fill = (255,0,0))
                img0 = cv2.cvtColor(np.array(img_pil),cv2.COLOR_RGB2BGR) 

            # Show results
            cv2.imshow('RealSense', img0)
            cv2.waitKey(5)
            if most_common_num != -1 :
                return most_common_num
            if time.time() - t0 >= 20:
                return 8
    finally:
       
        print(f'Number_Done. ({time.time() - t0:.3f}s)')

def detect_zhuixingtong(save_img=False,detect_number=0,cut_image_1=1.0,cut_image_2=1.0):
    most_common_num = -1
    
    # variable initialized here
    number_classified_result = deque([-1] * 10,maxlen=10)

    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device4)['model']).to(device4).eval()

    # Get names and colors
    names = model4.module.names if hasattr(model4, 'module') else model4.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
    # Run inference
    if device4.type != 'cpu':
        model4(torch.zeros(1, 3, imgsz4, imgsz4).to(device4).type_as(next(model4.parameters())))  # run once
    old_img_w = old_img_h = imgsz4
    old_img_b = 1

    t0 = time.time()  # Initialize t0

    try:
        while True:
            robot.send_data(0x21010130, -20000, 0)
            time.sleep(0.05)
            # Wait for a new frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert frame to numpy array
            
            img0 = np.asanyarray(color_frame.get_data())
            img0 = img0[:, 0:int(img0.shape[1] *cut_image_1 / cut_image_2)]
            HSV_img0 = cv2.cvtColor(img0,cv2.COLOR_BGR2HSV)
            # img0 = cv2.imread("gesture/gesture04.jpg")
            img = letterbox(img0, 640, 32)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            #img = img0.transpose(2, 0, 1)  # HWC to CHW
            img = torch.from_numpy(img).to(device4)
            img = img.half() if half4 else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            #print(img.ndimension())
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device4.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model4(img, augment=opt4.augment)[0]



            # Inference
            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                pred = model4(img, augment=opt4.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt4.conf_thres, opt4.iou_thres, classes=opt4.classes, agnostic=opt4.agnostic_nms)
            
	   
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    # Rescale boxes from img_size to img0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                     # Sort detections by confidence
                    det = det[det[:, 4].argsort(descending=True)]

                    # Keep only the highest confidence detection
                    det = det[:2]

                    # Only keep the highest confidence detection per class
                    filtered_det = []
                    for *xyxy, conf, cls in det:
                        filtered_det.append((xyxy, conf, cls))

                    result = -1
                    left_x = 10000
                    max_area = 0
                    # Write results
                    for *xyxy, conf, cls in filtered_det:
                        if conf < 0.15:
                            continue
                        if detect_number == 0:
                            if left_x > xyxy[0][0].item():
                                left_x = xyxy[0][0].item()
                                #result = int(cls)
                                zhuixingtong_center_x, zhuixingtong_center_y = int((xyxy[0][0] + xyxy[0][2]) // 2), int((xyxy[0][1] + xyxy[0][3]) // 2)
                                window_size = int((xyxy[0][2] - xyxy[0][0]) // 5)
                                half_window = int((xyxy[0][3] - xyxy[0][1]) // 5)
                                BGR_region = img0[zhuixingtong_center_y-half_window:zhuixingtong_center_y+half_window, zhuixingtong_center_x-window_size:zhuixingtong_center_x+window_size]
                                HSV_region = HSV_img0[zhuixingtong_center_y-half_window:zhuixingtong_center_y+half_window, zhuixingtong_center_x-window_size:zhuixingtong_center_x+window_size]
                                cv2.rectangle(img0, (zhuixingtong_center_x-window_size,zhuixingtong_center_y-half_window),(zhuixingtong_center_x+window_size,zhuixingtong_center_y+half_window),[0,0,255],1,lineType=cv2.LINE_AA)
                                BGR_mean_vals = np.mean(BGR_region,axis=(0,1))
                                mean_b, mean_g, mean_r = BGR_mean_vals
                                HSV_mean_vals = np.mean(HSV_region,axis=(0,1))
                                mean_h, mean_s, mean_v = HSV_mean_vals
                                print(mean_h/mean_v,(mean_h, mean_s, mean_v),(mean_b, mean_g, mean_r),conf)
                                if mean_r < 100:
                                    result = 2
                                else:
                                    if mean_g > 120:
                                        result = 3
                                    else:
                                        if mean_h / mean_v < 0.72: # or mean_g > 40:
                                            result = 1
                                        else:
                                            result = 0
                                label = "conical_cylinder"
                                # label = f'{names[int(cls)]} {conf:.2f}'
                                plot_one_box(xyxy[0], img0, label=label, color=colors[int(cls)], line_thickness=1)
                        else:
                            if max_area < (xyxy[0][2].item() - xyxy[0][0].item())**2:
                                max_area = (xyxy[0][2].item() - xyxy[0][0].item())**2
                                #result = int(cls)
                                zhuixingtong_center_x, zhuixingtong_center_y = int((xyxy[0][0] + xyxy[0][2]) // 2), int((xyxy[0][1] + xyxy[0][3]) // 2)
                                window_size = int((xyxy[0][2] - xyxy[0][0]) // 5)
                                half_window = int((xyxy[0][3] - xyxy[0][1]) // 5)
                                BGR_region = img0[zhuixingtong_center_y-half_window:zhuixingtong_center_y+half_window, zhuixingtong_center_x-window_size:zhuixingtong_center_x+window_size]
                                HSV_region = HSV_img0[zhuixingtong_center_y-half_window:zhuixingtong_center_y+half_window, zhuixingtong_center_x-window_size:zhuixingtong_center_x+window_size]
                                cv2.rectangle(img0, (zhuixingtong_center_x-window_size,zhuixingtong_center_y-half_window),(zhuixingtong_center_x+window_size,zhuixingtong_center_y+half_window),[0,0,255],1,lineType=cv2.LINE_AA)
                                BGR_mean_vals = np.mean(BGR_region,axis=(0,1))
                                mean_b, mean_g, mean_r = BGR_mean_vals
                                HSV_mean_vals = np.mean(HSV_region,axis=(0,1))
                                mean_h, mean_s, mean_v = HSV_mean_vals
                                print(mean_h/mean_v,(mean_h, mean_s, mean_v),(mean_b, mean_g, mean_r))
                                if mean_r < 100:
                                    result = 2
                                else:
                                    if mean_g > 120:
                                        result = 3
                                    else:
                                        #if mean_h / mean_v < 0.75:
                                        if mean_h / mean_v < 0.72:
                                            result = 1
                                        else:
                                            result = 0
                                label = "conical_cylinder"
                                # label = f'{names[int(cls)]} {conf:.2f}'
                                plot_one_box(xyxy[0], img0, label=label, color=colors[int(cls)], line_thickness=1)                             

                    if result != -1:
                        number_classified_result.append(result)
            
            if len(number_classified_result) > 0:
                img_pil = Image.fromarray(cv2.cvtColor(img0,cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                count = Counter(number_classified_result) 
                most_common_num, most_common_count = count.most_common(1)[0] 
                if most_common_num == 0:
                    draw.text((150,50),"红色区域",font = font,fill = (255,0,0)) 
                elif most_common_num == 1:
                    draw.text((150,50),"橙色区域",font = font,fill = (255,0,0)) 
                elif most_common_num == 2:
                    draw.text((150,50),"蓝色区域",font = font,fill = (255,0,0)) 
                elif most_common_num == 3:
                    draw.text((150,50),"黄色区域",font = font,fill = (255,0,0))
                img0 = cv2.cvtColor(np.array(img_pil),cv2.COLOR_RGB2BGR)
            

            # Show results
            cv2.imshow('RealSense', img0)
            cv2.waitKey(5)
            if most_common_num != -1:
                return most_common_num
            if time.time() - t0 >= 10:
                return 4
    finally:
        robot.send_data(0x21010D05, 0, 0)
        time.sleep(0.1)
        robot.send_data(0x21010C03, 0, 0)
        time.sleep(0.1)
        print(f'ZXTDone. ({time.time() - t0:.3f}s)')

def detect_ball(save_img=False):
    most_common_num = -1
    
    # variable initialized here
    ball_classified_result = deque([-1] * 10,maxlen=10)

    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device5)['model']).to(device5).eval()

    # Get names and colors
    names = model5.module.names if hasattr(model5, 'module') else model5.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
    # Run inference
    if device5.type != 'cpu':
        model5(torch.zeros(1, 3, imgsz5, imgsz5).to(device5).type_as(next(model5.parameters())))  # run once
    old_img_w = old_img_h = imgsz5
    old_img_b = 1

    t0 = time.time()  # Initialize t0

    try:
        while True:
            # Wait for a new frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert frame to numpy array
            
            img0 = np.asanyarray(color_frame.get_data())
            HSV_img0 = cv2.cvtColor(img0,cv2.COLOR_BGR2HSV)

            # img0 = cv2.imread("gesture/gesture04.jpg")
            img = letterbox(img0, 640, 32)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            #img = img0.transpose(2, 0, 1)  # HWC to CHW
            img = torch.from_numpy(img).to(device5)
            img = img.half() if half5 else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device5.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model5(img, augment=opt5.augment)[0]



            # Inference
            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                pred = model5(img, augment=opt5.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt5.conf_thres, opt5.iou_thres, classes=opt5.classes, agnostic=opt5.agnostic_nms)
            
	   
            result = -1
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    # Rescale boxes from img_size to img0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                     # Sort detections by confidence
                    det = det[det[:, 4].argsort(descending=True)]

                    # Keep only the highest confidence detection
                    det = det[:1]

                    # Only keep the highest confidence detection per class
                    seen_classes = {}
                    filtered_det = []
                    for *xyxy, conf, cls in det:
                        if cls.item() not in seen_classes:
                            seen_classes[cls.item()] = True
                            filtered_det.append((xyxy, conf, cls))

                    # Write results
                    
                    for *xyxy, conf, cls in filtered_det:
                        if conf > 0.25:
                            #gesture_classified_result.append(int(cls))
                            label = f'{conf:.2f}'
                            # label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy[0], img0, label=label, color=colors[int(cls)], line_thickness=1)
                            ball_center_x, ball_center_y = int((xyxy[0][0] + xyxy[0][2]) // 2), int((xyxy[0][1] + xyxy[0][3]) // 2)
                            window_size = int((xyxy[0][2] - xyxy[0][0])//10)
                            half_window = int((xyxy[0][3] - xyxy[0][1]) // 10)
                            region = img0[ball_center_y-half_window:ball_center_y+half_window, ball_center_x-window_size:ball_center_x+window_size]
                            HSV_region = HSV_img0[ball_center_y-half_window:ball_center_y+half_window, ball_center_x-window_size:ball_center_x+window_size]
                            cv2.rectangle(img0, (ball_center_x-window_size,ball_center_y-half_window),(ball_center_x+window_size,ball_center_y+half_window),[0,0,255],1,lineType=cv2.LINE_AA)
                            mean_vals = np.mean(region,axis=(0,1))
                            mean_b, mean_g, mean_r = mean_vals
                            HSV_mean_vals = np.mean(HSV_region,axis=(0,1))
                            mean_h, mean_s, mean_v = HSV_mean_vals

                            print(mean_b, mean_g, mean_r)
                            if mean_r < 150:
                                result = 2
                            else:
                                if mean_b > mean_g - 7 : #or (mean_g < 40 and mean_b < 40):
                                    result = 0
                                else:
                                   mean_b = mean_b * 253.0 / mean_r
                                   mean_g = mean_g * 253.0 / mean_r
                                   mean_r = mean_r * 253.0 / mean_r
                                   if mean_g > 100:
                                       result = 3
                                   else:
                                       result = 1
                            
                            
                            print("ball",(mean_h, mean_s, mean_v), (ball_center_x, ball_center_y, mean_b, mean_g, mean_r))
                else:
                    publish_cmd_vel(pub,0.1,0,0.1,0) #move left
                    
            ball_classified_result.append(result)
            if len(ball_classified_result) > 0:
                img_pil = Image.fromarray(cv2.cvtColor(img0,cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                count = Counter(ball_classified_result) 
                most_common_num, most_common_count = count.most_common(1)[0] 
                if most_common_num == 0:
                    draw.text((150,50),"红色球",font = font,fill = (255,0,0)) 
                elif most_common_num == 1:
                    draw.text((150,50),"橙色球",font = font,fill = (255,0,0)) 
                elif most_common_num == 2:
                    draw.text((150,50),"蓝色球",font = font,fill = (255,0,0)) 
                elif most_common_num == 3:
                    draw.text((150,50),"黄色球",font = font,fill = (255,0,0))
                img0 = cv2.cvtColor(np.array(img_pil),cv2.COLOR_RGB2BGR)
            

            # Show results
            cv2.imshow('RealSense', img0)
            cv2.waitKey(5)
            if most_common_num != -1:
                return most_common_num
            if time.time() - t0 >= 20:
                return 4
    finally:
        print(f'BallDone. ({time.time() - t0:.3f}s)')
        


play_lock = threading.Lock()

def pygame_play(file_path, volume):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.set_volume(volume)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
def play_mp3(file_path1, file_path2, file_path3):
    with play_lock:
        pygame_play(file_path1,1.0)
        pygame_play(file_path2,1.0)
        pygame_play(file_path3,1.0)   

if __name__ == '__main__':
    zhuixingtong_list = ["红色巡检区", "橙色巡检去", "蓝色巡检区", "黄色巡检区", "未知巡检区"]
    ball_list = ["红色球", "橙色球", "蓝色球", "黄色球", "未知球"]
    number_list = ["1号仪表盘", "2号仪表盘", "3号仪表盘", "4号仪表盘", "5号仪表盘", "6号仪表盘", "7号仪表盘", "8号仪表盘", "*号仪表盘"]
    status_list = ["显示偏低", "显示正常", "显示偏高", "显示未知"]
    area_mp3 = [
        os.path.join("voice","hongsexunjianqu.mp3"),
        os.path.join("voice","chengsexunjianqu.mp3"),
        os.path.join("voice","lansexunjianqu.mp3"),
        os.path.join("voice","huangsexunjianqu.mp3"),
        os.path.join("voice","weizhixunjianqu.mp3")
    ]
    number_mp3 = [
        os.path.join("voice","1.mp3"),
        os.path.join("voice","2.mp3"),
        os.path.join("voice","3.mp3"),
        os.path.join("voice","4.mp3"),
        os.path.join("voice","5.mp3"),
        os.path.join("voice","6.mp3"),
        os.path.join("voice","7.mp3"),
        os.path.join("voice","8.mp3"),
        os.path.join("voice","weizhiyibiaopan.mp3")    
    ]
    status_mp3 = [
        os.path.join("voice","xianshipiandi.mp3"),
        os.path.join("voice","xianshipiandi.mp3"),
        os.path.join("voice","xianshipiangao.mp3"),
        os.path.join("voice","xianshiweizhi.mp3")
    ]
    area_color = [-1, -1, -1, -1]
    area_danger = np.full((4,2),-1)
    ball_color = [-1, -1, -1, -1]

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    change_flag = ord('h')
    
    parser1 = argparse.ArgumentParser()
    parser1.add_argument('--weights', nargs='+', type=str, default='yolo_models/best_hand.pt', help='model.pt path(s)')
    parser1.add_argument('--source', type=str, default='datasets/images/test', help='source')  # file/folder, 0 for webcam
    parser1.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser1.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser1.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser1.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser1.add_argument('--view-img', action='store_true', help='display results')
    parser1.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser1.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser1.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser1.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser1.add_argument('--augment', action='store_true', help='augmented inference')
    parser1.add_argument('--update', action='store_true', help='update all models')
    parser1.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser1.add_argument('--name', default='exp', help='save results to project/name')
    parser1.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser1.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt1 = parser1.parse_args()
    print(opt1)
    
    source, weights, view_img, save_txt, imgsz, trace = opt1.source, opt1.weights, opt1.view_img, opt1.save_txt, opt1.img_size, not opt1.no_trace
    save_img = not opt1.nosave and not source.endswith('.txt')  # save inference images

    # Directories
    save_dir = Path(increment_path(Path(opt1.project) / opt1.name, exist_ok=opt1.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt1.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt1.img_size)

    if half:
        model.half()  # to FP16
    
    #find_dashboard/pointer/sign_model
    parser2 = argparse.ArgumentParser()
    parser2.add_argument('--weights', nargs='+', type=str, default='yolo_models/best_dashboard.pt', help='model.pt path(s)')
    parser2.add_argument('--source', type=str, default='datasets/images/test', help='source')  # file/folder, 0 for webcam
    parser2.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser2.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser2.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser2.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser2.add_argument('--view-img', action='store_true', help='display results')
    parser2.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser2.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser2.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser2.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser2.add_argument('--augment', action='store_true', help='augmented inference')
    parser2.add_argument('--update', action='store_true', help='update all models')
    parser2.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser2.add_argument('--name', default='exp', help='save results to project/name')
    parser2.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser2.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt2 = parser2.parse_args()
    print(opt2)

    source2, weights2, view_img2, save_txt2, imgsz2, trace2 = opt2.source, opt2.weights, opt2.view_img, opt2.save_txt, opt2.img_size, not opt2.no_trace
    save_img2 = not opt2.nosave and not source2.endswith('.txt')  # save inference images

    # Directories
    save_dir2 = Path(increment_path(Path(opt2.project) / opt2.name, exist_ok=opt2.exist_ok))  # increment run
    (save_dir2 / 'labels' if save_txt2 else save_dir2).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device2 = select_device(opt2.device)
    half2 = device2.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model2 = attempt_load(weights2, map_location=device2)  # load FP32 model
    stride2 = int(model2.stride.max())  # model stride
    imgsz2 = check_img_size(imgsz2, s=stride2)  # check img_size

    if trace2:
        model2 = TracedModel(model2, device2, opt2.img_size)

    if half2:
        model2.half()  # to FP16
    
    # classify number model
    parser3 = argparse.ArgumentParser()
    parser3.add_argument('--weights', nargs='+', type=str, default='yolo_model/best_number.pt', help='model.pt path(s)')
    parser3.add_argument('--source', type=str, default='datasets/images/test', help='source')  # file/folder, 0 for webcam
    parser3.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser3.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser3.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser3.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser3.add_argument('--view-img', action='store_true', help='display results')
    parser3.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser3.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser3.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser3.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser3.add_argument('--augment', action='store_true', help='augmented inference')
    parser3.add_argument('--update', action='store_true', help='update all models')
    parser3.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser3.add_argument('--name', default='exp', help='save results to project/name')
    parser3.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser3.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt3 = parser3.parse_args()
    print(opt3)

    source3, weights3, view_img3, save_txt3, imgsz3, trace3 = opt3.source, opt3.weights, opt3.view_img, opt3.save_txt, opt3.img_size, not opt3.no_trace
    save_img3 = not opt3.nosave and not source3.endswith('.txt')  # save inference images

    # Directories
    save_dir3 = Path(increment_path(Path(opt3.project) / opt3.name, exist_ok=opt3.exist_ok))  # increment run
    (save_dir3 / 'labels' if save_txt3 else save_dir3).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device3 = select_device(opt3.device)
    half3 = device3.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model3 = attempt_load(weights3, map_location=device3)  # load FP32 model
    stride3 = int(model3.stride.max())  # model stride
    imgsz3 = check_img_size(imgsz3, s=stride3)  # check img_size

    if trace3:
        model3 = TracedModel(model3, device3, opt3.img_size)

    if half3:
        model3.half()  # to FP16

    # classify zhuixingtong model
    parser4 = argparse.ArgumentParser()
    parser4.add_argument('--weights', nargs='+', type=str, default='yolo_models/best_zxt.pt', help='model.pt path(s)')
    parser4.add_argument('--source', type=str, default='datasets/images/test', help='source')  # file/folder, 0 for webcam
    parser4.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser4.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser4.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser4.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser4.add_argument('--view-img', action='store_true', help='display results')
    parser4.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser4.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser4.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser4.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser4.add_argument('--augment', action='store_true', help='augmented inference')
    parser4.add_argument('--update', action='store_true', help='update all models')
    parser4.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser4.add_argument('--name', default='exp', help='save results to project/name')
    parser4.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser4.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt4 = parser4.parse_args()
    print(opt4)

    source4, weights4, view_img4, save_txt4, imgsz4, trace4 = opt4.source, opt4.weights, opt4.view_img, opt4.save_txt, opt4.img_size, not opt4.no_trace
    save_img4 = not opt4.nosave and not source4.endswith('.txt')  # save inference images

    # Directories
    save_dir4 = Path(increment_path(Path(opt4.project) / opt4.name, exist_ok=opt4.exist_ok))  # increment run
    (save_dir4 / 'labels' if save_txt4 else save_dir4).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device4 = select_device(opt4.device)
    half4 = device4.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model4 = attempt_load(weights4, map_location=device4)  # load FP32 model
    stride4 = int(model4.stride.max())  # model stride
    imgsz4 = check_img_size(imgsz4, s=stride4)  # check img_size

    if trace4:
        model4 = TracedModel(model4, device4, opt4.img_size)

    if half4:
        model4.half()  # to FP16

    # classify ball model
    parser5 = argparse.ArgumentParser()
    parser5.add_argument('--weights', nargs='+', type=str, default='yolo_models/best_ball.pt', help='model.pt path(s)')
    parser5.add_argument('--source', type=str, default='datasets/images/test', help='source')  # file/folder, 0 for webcam
    parser5.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser5.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser5.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser5.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser5.add_argument('--view-img', action='store_true', help='display results')
    parser5.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser5.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser5.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser5.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser5.add_argument('--augment', action='store_true', help='augmented inference')
    parser5.add_argument('--update', action='store_true', help='update all models')
    parser5.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser5.add_argument('--name', default='exp', help='save results to project/name')
    parser5.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser5.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt5 = parser5.parse_args()
    print(opt5)

    source5, weights5, view_img5, save_txt5, imgsz5, trace5 = opt5.source, opt5.weights, opt5.view_img, opt5.save_txt, opt5.img_size, not opt5.no_trace
    save_img5 = not opt5.nosave and not source5.endswith('.txt')  # save inference images

    # Directories
    save_dir5 = Path(increment_path(Path(opt5.project) / opt5.name, exist_ok=opt5.exist_ok))  # increment run
    (save_dir5 / 'labels' if save_txt5 else save_dir5).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device5 = select_device(opt5.device)
    half5 = device5.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model5 = attempt_load(weights5, map_location=device5)  # load FP32 model
    stride5 = int(model5.stride.max())  # model stride
    imgsz5 = check_img_size(imgsz5, s=stride5)  # check img_size

    if trace5:
        model5 = TracedModel(model5, device5, opt5.img_size)

    if half5:
        model5.half()  # to FP16

    ctx = rs.context()
    while True:
        devices = ctx.query_devices()
        if len(devices) != 0:
            break
        else:
            print("没有检测到RealSense设备。")
    # Initialize RealSense camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    time.sleep(0.05)
    
    robot = ControlRobot()
    pub = publisher_init() #init pub

    start_flag = None
    while True:
        # Wait for a new frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame is None:
            continue
        img_init = np.asanyarray(color_frame.get_data())
        cv2.imshow('Init', img_init)
        start_flag = cv2.waitKey(5) & 0xFF
        if start_flag == ord('s'):
            break    
    cv2.destroyAllWindows()
    robot.send_data(0x21010C02, 0, 0) #shou dong mo shi
    time.sleep(0.05)
    
    begin_time = time.time()
    robot.stand_up()
    print("stand_up")
    time.sleep(0.5)  # waiting for standing up
    
    gesture = detect_gesture()
    if gesture == 0:
        robot.forward(0.8)
    elif gesture == 1:
        robot.backward(0.8)
    elif gesture == 2:
        robot.twist(3)
        time.sleep(0.05)
        robot.stop()
    elif gesture == 3:
        robot.move_left(0.8)
    elif gesture == 4:
        robot.move_right(0.8)
    elif gesture == 5:
        robot.turn_left1(0.5)
    elif gesture == 6:
        robot.turn_right1(0.5)
    
    
    if gesture == 2:
        time.sleep(4)
        robot.stand_up()
        time.sleep(1)
        robot.stand_up()
    else:
        time.sleep(1)
    time.sleep(0.5)
    gesture = detect_gesture()
    if gesture == 0:
        robot.forward(0.8)
    elif gesture == 1:
        robot.backward(0.8)
    elif gesture == 2:
        robot.twist(4)
        time.sleep(0.05)
        robot.stop()
    elif gesture == 3:
        robot.move_left(0.8)
    elif gesture == 4:
        robot.move_right(0.8)
    elif gesture == 5:
        robot.turn_left1(0.5)
    elif gesture == 6:
        robot.turn_right1(0.5)
        
    if gesture == 2:
        time.sleep(4)
        robot.stand_up()
        time.sleep(1)
        robot.stand_up()
    else:
        time.sleep(1)   
    time.sleep(0.5)
    gesture = detect_gesture()
    if gesture == 0:
        robot.forward(0.8)
    elif gesture == 1:
        robot.backward(0.8)
    elif gesture == 2:
        robot.twist(5)
        time.sleep(0.05)
        robot.stop()
    elif gesture == 3:
        robot.move_left(0.8)
    elif gesture == 4:
        robot.move_right(0.8)
    elif gesture == 5:
        robot.turn_left1(0.5)
    elif gesture == 6:
        robot.turn_right1(0.5)
    time.sleep(12)
    
    if gesture == 2:
        robot.stand_up()
        time.sleep(1)
        robot.stand_up()    
        robot.forward(2.4)
        time.sleep(1)
    else:
        robot.forward(2.0)
        time.sleep(1)

    robot.send_data(0x21010407, 0, 0) #gao tai tui
    time.sleep(0.5)

    robot.forward(6)  # walking forward for 4s
    print("forward")
    time.sleep(0.5)

    robot.send_data(0x21010300, 0, 0) #zheng chang bu fa
    time.sleep(0.5)
    
    robot.send_data(0x21010C03, 0, 0) #zi zhu mo shi
    time.sleep(0.05)
    
    publish_cmd_vel(pub,-1,0,0,0) 
    time.sleep(2)
    publish_cmd_vel(pub,0.65,0.5,0,-1) #turn right
    time.sleep(0.1)
    publish_cmd_vel(pub,1.3,1,0,0) #forward
    time.sleep(0.5)
    publish_cmd_vel(pub,0.55,0.5,0,1) #turn left and forward
    time.sleep(0.1)

    move_to_dashboard_center(False, pub, v_x=-0.2, v_y=0.1) ###################################### dashboard1
    #time.sleep(3)
    
    dashboard_idx = detect_dashboard()
    number_idx = detect_number()
    zhuixingtong_idx = detect_zhuixingtong(detect_number=1)
    area_color[0] = zhuixingtong_idx
    if dashboard_idx == 1:
        print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_idx], "状态正常,", status_list[dashboard_idx])
    else:
        area_danger[0][0] = 1
        threading.Thread(target=play_mp3, args=(area_mp3[zhuixingtong_idx],number_mp3[number_idx],status_mp3[dashboard_idx])).start()
        print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_idx], "状态异常,", status_list[dashboard_idx])


    publish_cmd_vel(pub,1.0,0,0,1) #turn left
    dashboard_idx = detect_dashboard() ######dashboard 8
    number_idx = detect_number()
    number_8 = number_idx
    zhuixingtong_idx = detect_zhuixingtong()
    cv2.destroyAllWindows()
    area_color[3] = zhuixingtong_idx
    
    if dashboard_idx == 1:
        print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_idx], "状态正常,", status_list[dashboard_idx])
    else:
        area_danger[3][1] = dashboard_idx
        # playsound(area_mp3[zhuixingtong_idx])
        # playsound(number_mp3[number_idx])
        # playsound(status_mp3[dashboard_idx])
        if zhuixingtong_idx != 0:
            threading.Thread(target=play_mp3, args=(area_mp3[zhuixingtong_idx],number_mp3[number_idx],status_mp3[dashboard_idx])).start()
            print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_idx], "状态异常,", status_list[dashboard_idx])
    
    publish_cmd_vel(pub,3.65,0,0,1) #turn left
    ball_color[0] = detect_ball()    #ball1
    time.sleep(1)
    publish_cmd_vel(pub,0.7,0,0,1) #turn left
    time.sleep(0.1)
    publish_cmd_vel(pub,3,0.5,0,0) #forward
    time.sleep(0.1)
    publish_cmd_vel(pub,1.0,0,0,1) #turn left
    time.sleep(0.1)
    align_with_white_line(robot, pub, pipeline) #xunji
    publish_cmd_vel(pub,5.5,1,0,0) #forward
    time.sleep(0.1)
    publish_cmd_vel(pub,1.65,0,0,1) #turn left
    time.sleep(0.1)
    move_to_dashboard_center(False, pub) ###################################### dashboard3
    #time.sleep(3)
    
    dashboard_idx = detect_dashboard()
    number_idx = detect_number()
    zhuixingtong_idx = detect_zhuixingtong(detect_number=1)
    area_color[1] = zhuixingtong_idx
    if dashboard_idx == 1:
        print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_idx], "状态正常,", status_list[dashboard_idx])
    else:
        area_danger[1][0] = 1
        threading.Thread(target=play_mp3, args=(area_mp3[zhuixingtong_idx],number_mp3[number_idx],status_mp3[dashboard_idx])).start()
        print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_idx], "状态异常,", status_list[dashboard_idx])


    publish_cmd_vel(pub,1,0,0,1) #turn left
    dashboard_idx = detect_dashboard() ######dashboard 2
    number_idx = detect_number()
    zhuixingtong_idx = area_color[0]
    cv2.destroyAllWindows()

    
    if dashboard_idx == 1:
        print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_idx], "状态正常,", status_list[dashboard_idx])
    else:
        area_danger[0][1] = 1
        threading.Thread(target=play_mp3, args=(area_mp3[zhuixingtong_idx],number_mp3[number_idx],status_mp3[dashboard_idx])).start()
        print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_idx], "状态异常,", status_list[dashboard_idx])
    
    robot.send_data(0x21010D05, 0, 0)
    time.sleep(0.1)
    robot.send_data(0x21010C03, 0, 0)
    time.sleep(0.1)    
    publish_cmd_vel(pub,3.2,0,0,1) #turn left
    ball_color[1] = detect_ball()    #ball2
    time.sleep(1.05)
    publish_cmd_vel(pub,0.8,0,0,1) #turn left
    time.sleep(0.15)
    publish_cmd_vel(pub,3.0,0.5,0,0) #forward
    time.sleep(0.1)
    publish_cmd_vel(pub,1.3,0,0,1) #turn left
    time.sleep(0.1)
    align_with_white_line(robot, pub, pipeline) #xunji
    time.sleep(0.1)
    publish_cmd_vel(pub,5.5,1,0,0) #forward
    time.sleep(0.1)
    publish_cmd_vel(pub,1.65,0,0,1) #turn left
    time.sleep(0.1)
    move_to_dashboard_center(False, pub) ###################################### dashboard5
    #time.sleep(3)
    
    dashboard_idx = detect_dashboard()
    number_idx = detect_number()
    zhuixingtong_idx = detect_zhuixingtong(detect_number=1)
    area_color[2] = zhuixingtong_idx
    if dashboard_idx == 1:
        print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_idx], "状态正常,", status_list[dashboard_idx])
    else:
        area_danger[2][0] = 1
        threading.Thread(target=play_mp3, args=(area_mp3[zhuixingtong_idx],number_mp3[number_idx],status_mp3[dashboard_idx])).start()
        print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_idx], "状态异常,", status_list[dashboard_idx])


    publish_cmd_vel(pub,1,0,0,1) #turn left
    dashboard_idx = detect_dashboard() ######dashboard 4
    number_idx = detect_number()
    #zhuixingtong_idx = detect_zhuixingtong()
    zhuixingtong_idx = area_color[1]
    cv2.destroyAllWindows()
    if dashboard_idx == 1:
        print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_idx], "状态正常,", status_list[dashboard_idx])
    else:
        area_danger[1][1] = 1
        threading.Thread(target=play_mp3, args=(area_mp3[zhuixingtong_idx],number_mp3[number_idx],status_mp3[dashboard_idx])).start()
        print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_idx], "状态异常,", status_list[dashboard_idx])
    
    robot.send_data(0x21010D05, 0, 0)
    time.sleep(0.1)
    robot.send_data(0x21010C03, 0, 0)
    time.sleep(0.1)   
    publish_cmd_vel(pub,3.2,0,0,1) #turn left
    ball_color[2] = detect_ball()    #ball3
    time.sleep(1.05)
    publish_cmd_vel(pub,0.95,0,0,1) #turn left
    time.sleep(0.1)
    publish_cmd_vel(pub,3,0.5,0,0) #forward
    time.sleep(0.1)
    publish_cmd_vel(pub,1.15,0,0,1) #turn left 
    time.sleep(0.1)
    align_with_white_line(robot, pub, pipeline) #xunji
    publish_cmd_vel(pub,5,1,0,0) #forward
    time.sleep(0.1)
    publish_cmd_vel(pub,1.65,0,0,1) #turn left
    time.sleep(0.1)
    move_to_dashboard_center(False, pub) ###################################### dashboard7
    #time.sleep(3)
    
    is_play_dashboard8 = 1
    dashboard_idx = detect_dashboard()
    number_idx = detect_number()
    if area_color[3] != 0:
        zhuixingtong_idx = area_color[3]
    else:
        zhuixingtong_idx = detect_zhuixingtong(detect_number=1)
        area_color[3] = zhuixingtong_idx
        is_play_dashboard8 = 0
    cv2.destroyAllWindows()

    if dashboard_idx == 1:
        print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_idx], "状态正常,", status_list[dashboard_idx])
    else:
        area_danger[3][0] = 1
        threading.Thread(target=play_mp3, args=(area_mp3[zhuixingtong_idx],number_mp3[number_idx],status_mp3[dashboard_idx])).start()
        print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_idx], "状态异常,", status_list[dashboard_idx])
    
    if is_play_dashboard8 == 0 and area_danger[3][1] != 1: #########dashboard8
        threading.Thread(target=play_mp3, args=(area_mp3[zhuixingtong_idx],number_mp3[number_8],status_mp3[area_danger[3][1]])).start()
        print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_8], "状态异常,", status_list[area_danger[3][1]])




    robot.send_data(0x21010D05, 0, 0)
    time.sleep(0.1)
    robot.send_data(0x21010C03, 0, 0)
    time.sleep(0.1)       
    publish_cmd_vel(pub,1,0,0,1) #turn left
    dashboard_idx = detect_dashboard() ######dashboard 6
    number_idx = detect_number()
    zhuixingtong_idx = area_color[2]
    cv2.destroyAllWindows()
    if dashboard_idx == 1:
        print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_idx], "状态正常,", status_list[dashboard_idx])
    else:
        area_danger[2][1] = 1
        threading.Thread(target=play_mp3, args=(area_mp3[zhuixingtong_idx],number_mp3[number_idx],status_mp3[dashboard_idx])).start()
        print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_idx], "状态异常,", status_list[dashboard_idx])
    
    robot.send_data(0x21010D05, 0, 0)
    time.sleep(0.1)
    robot.send_data(0x21010C03, 0, 0)
    time.sleep(0.1)      
    publish_cmd_vel(pub,3.2,0,0,1) #turn left
    time.sleep(0.1)    #ball4
    ball_color[3] = detect_ball()
    danger_idx = list(set(np.where(area_danger != np.full((4,2),-1))[0]))
    print('danger_idx',danger_idx)
    print('area_danger',area_danger)
    print('area_color',area_color)
    print('ball_color',ball_color)
    
    if area_color[danger_idx[0]] == ball_color[3] or area_color[danger_idx[1]] == ball_color[3]:
        move_to_ball_center(False,pub)
        time.sleep(0.1)
        publish_cmd_vel(pub,0.5,0.5,0,0) #forward
        time.sleep(0.1)
        publish_cmd_vel(pub,0.5,-0.5,0,0) #backward
        time.sleep(0.1)
        if area_color[danger_idx[0]] == ball_color[0] or area_color[danger_idx[1]] == ball_color[0]:
            publish_cmd_vel(pub,1.8,0,0,1) #turn left
            time.sleep(0.1)
            align_with_white_line(robot, pub, pipeline) #xunji
            publish_cmd_vel(pub,7.1,0.5,0,0) #forward
            time.sleep(0.1)
            publish_cmd_vel(pub,0.6,0,0,-1) #turn right
            move_to_ball_center(False,pub)
            time.sleep(0.1)
            publish_cmd_vel(pub,2.5,0.5,0,0) #forward
            time.sleep(0.1)
            publish_cmd_vel(pub,2.5,-0.5,0,0) #backward
        elif area_color[danger_idx[0]] == ball_color[2] or area_color[danger_idx[1]] == ball_color[2]:
            publish_cmd_vel(pub,3.2,0,0,-1) #turn right
            time.sleep(0.1)
            align_with_white_line(robot, pub, pipeline, 497, 65, -0.1) #xunji
            time.sleep(0.1)
            publish_cmd_vel(pub,6.0,0.5,0,0) #forward
            time.sleep(0.1)
            publish_cmd_vel(pub,0.5,0,0,1) #turn left
            move_to_ball_center(False,pub)
            time.sleep(0.1)
            publish_cmd_vel(pub,0.5,0.5,0,0) #forward
            time.sleep(0.1)
            publish_cmd_vel(pub,-0.5,-0.5,0,0) #backward
            time.sleep(0.1)
        else:
            publish_cmd_vel(pub,1.8,0,0,1) #turn left
            time.sleep(0.1)
            align_with_white_line(robot, pub, pipeline) #xunji
            time.sleep(0.1)
            publish_cmd_vel(pub,7.5,0.5,0,0) #forward
            time.sleep(0.1)
            publish_cmd_vel(pub,1.5,0,0,1) #turn left
            time.sleep(0.1)
            align_with_white_line(robot, pub, pipeline) #xunji
            time.sleep(0.1)
            publish_cmd_vel(pub,6.0,0.5,0,0) #forward
            time.sleep(0.1)
            publish_cmd_vel(pub,0.6,0,0,-1) #turn right
            move_to_ball_center(False,pub)
            time.sleep(0.1)
            publish_cmd_vel(pub,1,0.5,0,0) #forward
            time.sleep(0.1)
            publish_cmd_vel(pub,1,-0.5,0,0) #backward
            time.sleep(0.1)
    elif area_color[danger_idx[0]] == ball_color[0] or area_color[danger_idx[1]] == ball_color[0]:
        time.sleep(1)
        publish_cmd_vel(pub,0.8,0,0,1) #turn left
        time.sleep(0.1)
        publish_cmd_vel(pub,3,0.5,0,0) #forward
        time.sleep(0.1)
        publish_cmd_vel(pub,1.1,0,0,1) #turn left 
        time.sleep(0.1)
        align_with_white_line(robot, pub, pipeline) #xunji
        time.sleep(0.1)
        publish_cmd_vel(pub,6,1,0,0) #forward
        time.sleep(0.1)
        publish_cmd_vel(pub,0.6,0,0,-1) #turn right
        time.sleep(0.1)
        move_to_ball_center(False,pub)
        time.sleep(0.1)
        publish_cmd_vel(pub,2.5,0.5,0,0) #forward
        time.sleep(0.1)
        publish_cmd_vel(pub,2.5,-0.5,0,0) #backward
        time.sleep(0.1)
        publish_cmd_vel(pub,2.2,0.3,0,1) #turn left
        time.sleep(0.1)
        align_with_white_line(robot, pub, pipeline) #xunji
        time.sleep(0.1)
        publish_cmd_vel(pub,6,1,0,0) #forward
        time.sleep(0.1)
        if area_color[danger_idx[0]] == ball_color[1] or area_color[danger_idx[1]] == ball_color[1]:
            time.sleep(0.1)
            publish_cmd_vel(pub,0.6,0,0,-1) #turn right
            time.sleep(0.1)
            move_to_ball_center(False,pub)
            time.sleep(0.1)
            publish_cmd_vel(pub,2.5,0.5,0,0) #forward
            time.sleep(0.1)
            publish_cmd_vel(pub,2.5,-0.5,0,0) #backward
        else:
            publish_cmd_vel(pub,0.6,0,0,1) #turn left
            time.sleep(0.1)
            publish_cmd_vel(pub,3.0,0.5,0,0) #forward
            time.sleep(0.1)
            publish_cmd_vel(pub,1.1,0,0,1) #turn left 
            time.sleep(0.1)
            align_with_white_line(robot, pub, pipeline) #xunji
            time.sleep(0.1)
            publish_cmd_vel(pub,6,1,0,0) #forward
            time.sleep(0.1)
            publish_cmd_vel(pub,0.6,0,0,-1) #turn right
            time.sleep(0.1)
            move_to_ball_center(False,pub)
            time.sleep(0.1)
            publish_cmd_vel(pub,0.5,0.5,0,0) #forward
            time.sleep(0.1)
            publish_cmd_vel(pub,0.5,-0.5,0,0) #backward
            time.sleep(0.1)
    else:
        time.sleep(1)
        publish_cmd_vel(pub,3,0,0,-1) #turn right
        time.sleep(0.1)
        publish_cmd_vel(pub,1,-0.5,0,0) #backward
        align_with_white_line(robot, pub, pipeline, 497, 65, -0.1) #xunji
        time.sleep(0.1)
        publish_cmd_vel(pub,6,1,0,0) #forward
        time.sleep(0.1)
        publish_cmd_vel(pub,0.6,0,0,1) #turn left
        time.sleep(0.1)
        move_to_ball_center(False,pub)
        time.sleep(0.1)
        publish_cmd_vel(pub,0.5,0.5,0,0) #forward
        time.sleep(0.1)
        publish_cmd_vel(pub,0.5,-0.5,0,0) #backward
        time.sleep(0.1)
        publish_cmd_vel(pub,2.2,0.3,0,-1) #turn right
        time.sleep(0.1)
        align_with_white_line(robot, pub, pipeline, 497, 65, -0.1)  #xunji
        time.sleep(0.1)
        publish_cmd_vel(pub,6.5,1,0,0) #forward
        time.sleep(0.1)
        publish_cmd_vel(pub,0.6,0,0,1) #turn left
        time.sleep(0.1)
        move_to_ball_center(False,pub)
        time.sleep(0.1)
        publish_cmd_vel(pub,1,0.5,0,0) #forward
        time.sleep(0.1)
        publish_cmd_vel(pub,1,-0.5,0,0) #backward
    print("time_all:", time.time() - begin_time)


    area_color = [-1, -1, -1, -1]
    area_danger = np.full((4,2),-1)
    ball_color = [-1, -1, -1, -1]    
    robot.send_data(0x21010C02, 0, 0) #shou dong mo shi
    time.sleep(2)
    robot.stand_up()
    
    start_flag = None
    while True:
        # Wait for a new frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame is None:
            continue
        img_init = np.asanyarray(color_frame.get_data())
        cv2.imshow('Init', img_init)
        start_flag = cv2.waitKey(5) & 0xFF
        if start_flag == ord('s'):
            break    
    cv2.destroyAllWindows()


    
    begin_time = time.time()
    robot.stand_up()
    print("stand_up")
    time.sleep(0.5)  # waiting for standing up
    
    gesture = detect_gesture()
    if gesture == 0:
        robot.forward(0.8)
    elif gesture == 1:
        robot.backward(0.8)
    elif gesture == 2:
        robot.twist(3)
        time.sleep(0.05)
        robot.stop()
    elif gesture == 3:
        robot.move_left(0.8)
    elif gesture == 4:
        robot.move_right(0.8)
    elif gesture == 5:
        robot.turn_left1(0.5)
    elif gesture == 6:
        robot.turn_right1(0.5)
    
    
    if gesture == 2:
        time.sleep(4)
        robot.stand_up()
        time.sleep(1)
        robot.stand_up()
    else:
        time.sleep(1)
    time.sleep(0.5)
    gesture = detect_gesture()
    if gesture == 0:
        robot.forward(0.8)
    elif gesture == 1:
        robot.backward(0.8)
    elif gesture == 2:
        robot.twist(4)
        time.sleep(0.05)
        robot.stop()
    elif gesture == 3:
        robot.move_left(0.8)
    elif gesture == 4:
        robot.move_right(0.8)
    elif gesture == 5:
        robot.turn_left1(0.5)
    elif gesture == 6:
        robot.turn_right1(0.5)
        
    if gesture == 2:
        time.sleep(4)

        robot.stand_up()
        time.sleep(1)
        robot.stand_up()
    else:
        time.sleep(1)   
    time.sleep(0.5)
    gesture = detect_gesture()
    if gesture == 0:
        robot.forward(0.8)
    elif gesture == 1:
        robot.backward(0.8)
    elif gesture == 2:
        robot.twist(5)
        time.sleep(0.05)
        robot.stop()
    elif gesture == 3:
        robot.move_left(0.8)
    elif gesture == 4:
        robot.move_right(0.8)
    elif gesture == 5:
        robot.turn_left1(0.5)
    elif gesture == 6:
        robot.turn_right1(0.5)
    time.sleep(12)
    
    if gesture == 2:
        robot.stand_up()
        time.sleep(1)
        robot.stand_up()    
        robot.forward(2.4)
        time.sleep(1)
    else:
        robot.forward(2.0)
        time.sleep(1)

    robot.send_data(0x21010407, 0, 0) #gao tai tui
    time.sleep(0.5)

    robot.forward(6)  # walking forward for 4s
    print("forward")
    time.sleep(0.5)

    robot.send_data(0x21010300, 0, 0) #zheng chang bu fa
    time.sleep(0.5)
    
    robot.send_data(0x21010C03, 0, 0) #zi zhu mo shi
    time.sleep(0.05)
    
    publish_cmd_vel(pub,-1,0,0,0) 
    time.sleep(2)
    publish_cmd_vel(pub,0.65,0.5,0,-1) #turn right
    time.sleep(0.1)
    publish_cmd_vel(pub,1.3,1,0,0) #forward
    time.sleep(0.5)
    publish_cmd_vel(pub,0.55,0.5,0,1) #turn left and forward
    time.sleep(0.1)

    move_to_dashboard_center(False, pub, v_x=-0.2, v_y=0.1) ###################################### dashboard1
    #time.sleep(3)
    
    dashboard_idx = detect_dashboard()
    number_idx = detect_number()
    zhuixingtong_idx = detect_zhuixingtong(detect_number=1)
    area_color[0] = zhuixingtong_idx
    if dashboard_idx == 1:
        print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_idx], "状态正常,", status_list[dashboard_idx])
    else:
        area_danger[0][0] = 1
        threading.Thread(target=play_mp3, args=(area_mp3[zhuixingtong_idx],number_mp3[number_idx],status_mp3[dashboard_idx])).start()
        print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_idx], "状态异常,", status_list[dashboard_idx])


    publish_cmd_vel(pub,1.0,0,0,1) #turn left
    dashboard_idx = detect_dashboard() ######dashboard 8
    number_idx = detect_number()
    number_8 = number_idx
    zhuixingtong_idx = detect_zhuixingtong()
    cv2.destroyAllWindows()
    area_color[3] = zhuixingtong_idx
    
    if dashboard_idx == 1:
        print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_idx], "状态正常,", status_list[dashboard_idx])
    else:
        area_danger[3][1] = dashboard_idx
        # playsound(area_mp3[zhuixingtong_idx])
        # playsound(number_mp3[number_idx])
        # playsound(status_mp3[dashboard_idx])
        if zhuixingtong_idx != 0:
            threading.Thread(target=play_mp3, args=(area_mp3[zhuixingtong_idx],number_mp3[number_idx],status_mp3[dashboard_idx])).start()
            print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_idx], "状态异常,", status_list[dashboard_idx])
    
    publish_cmd_vel(pub,3.65,0,0,1) #turn left
    ball_color[0] = detect_ball()    #ball1
    time.sleep(1)
    publish_cmd_vel(pub,0.7,0,0,1) #turn left
    time.sleep(0.1)
    publish_cmd_vel(pub,3,0.5,0,0) #forward
    time.sleep(0.1)
    publish_cmd_vel(pub,1,0,0,1) #turn left
    time.sleep(0.1)
    align_with_white_line(robot, pub, pipeline) #xunji
    publish_cmd_vel(pub,5.5,1,0,0) #forward
    time.sleep(0.1)
    publish_cmd_vel(pub,1.65,0,0,1) #turn left

    time.sleep(0.1)
    move_to_dashboard_center(False, pub) ###################################### dashboard3
    #time.sleep(3)
    
    dashboard_idx = detect_dashboard()
    number_idx = detect_number()
    zhuixingtong_idx = detect_zhuixingtong(detect_number=1)
    area_color[1] = zhuixingtong_idx
    if dashboard_idx == 1:
        print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_idx], "状态正常,", status_list[dashboard_idx])
    else:
        area_danger[1][0] = 1
        threading.Thread(target=play_mp3, args=(area_mp3[zhuixingtong_idx],number_mp3[number_idx],status_mp3[dashboard_idx])).start()
        print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_idx], "状态异常,", status_list[dashboard_idx])


    publish_cmd_vel(pub,1,0,0,1) #turn left
    dashboard_idx = detect_dashboard() ######dashboard 2
    number_idx = detect_number()
    zhuixingtong_idx = area_color[0]
    cv2.destroyAllWindows()

    
    if dashboard_idx == 1:
        print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_idx], "状态正常,", status_list[dashboard_idx])
    else:
        area_danger[0][1] = 1
        threading.Thread(target=play_mp3, args=(area_mp3[zhuixingtong_idx],number_mp3[number_idx],status_mp3[dashboard_idx])).start()
        print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_idx], "状态异常,", status_list[dashboard_idx])
    
    robot.send_data(0x21010D05, 0, 0)
    time.sleep(0.1)
    robot.send_data(0x21010C03, 0, 0)
    time.sleep(0.1)    
    publish_cmd_vel(pub,3.2,0,0,1) #turn left
    ball_color[1] = detect_ball()    #ball2
    time.sleep(1.05)
    publish_cmd_vel(pub,0.7,0,0,1) #turn left
    time.sleep(0.15)
    publish_cmd_vel(pub,3.0,0.5,0,0) #forward
    time.sleep(0.1)
    publish_cmd_vel(pub,1.3,0,0,1) #turn left
    time.sleep(0.1)
    align_with_white_line(robot, pub, pipeline) #xunji
    time.sleep(0.1)
    publish_cmd_vel(pub,5.5,1,0,0) #forward
    time.sleep(0.1)
    publish_cmd_vel(pub,1.65,0,0,1) #turn left
    time.sleep(0.1)
    move_to_dashboard_center(False, pub) ###################################### dashboard5
    #time.sleep(3)
    
    dashboard_idx = detect_dashboard()
    number_idx = detect_number()
    zhuixingtong_idx = detect_zhuixingtong(detect_number=1)
    area_color[2] = zhuixingtong_idx
    if dashboard_idx == 1:
        print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_idx], "状态正常,", status_list[dashboard_idx])
    else:
        area_danger[2][0] = 1
        threading.Thread(target=play_mp3, args=(area_mp3[zhuixingtong_idx],number_mp3[number_idx],status_mp3[dashboard_idx])).start()
        print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_idx], "状态异常,", status_list[dashboard_idx])


    publish_cmd_vel(pub,1,0,0,1) #turn left
    dashboard_idx = detect_dashboard() ######dashboard 4
    number_idx = detect_number()
    #zhuixingtong_idx = detect_zhuixingtong()
    zhuixingtong_idx = area_color[1]
    cv2.destroyAllWindows()
    if dashboard_idx == 1:
        print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_idx], "状态正常,", status_list[dashboard_idx])
    else:
        area_danger[1][1] = 1
        threading.Thread(target=play_mp3, args=(area_mp3[zhuixingtong_idx],number_mp3[number_idx],status_mp3[dashboard_idx])).start()
        print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_idx], "状态异常,", status_list[dashboard_idx])
    
    robot.send_data(0x21010D05, 0, 0)
    time.sleep(0.1)

    robot.send_data(0x21010C03, 0, 0)
    time.sleep(0.1)   
    publish_cmd_vel(pub,3.2,0,0,1) #turn left
    ball_color[2] = detect_ball()    #ball3
    time.sleep(1.05)
    publish_cmd_vel(pub,0.95,0,0,1) #turn left
    time.sleep(0.1)
    publish_cmd_vel(pub,3,0.5,0,0) #forward
    time.sleep(0.1)
    publish_cmd_vel(pub,1.15,0,0,1) #turn left 
    time.sleep(0.1)
    align_with_white_line(robot, pub, pipeline) #xunji
    publish_cmd_vel(pub,5,1,0,0) #forward
    time.sleep(0.1)
    publish_cmd_vel(pub,1.65,0,0,1) #turn left
    time.sleep(0.1)
    move_to_dashboard_center(False, pub) ###################################### dashboard7
    #time.sleep(3)
    
    is_play_dashboard8 = 1
    dashboard_idx = detect_dashboard()
    number_idx = detect_number()
    if area_color[3] != 0:
        zhuixingtong_idx = area_color[3]
    else:
        zhuixingtong_idx = detect_zhuixingtong(detect_number=1)
        area_color[3] = zhuixingtong_idx
        is_play_dashboard8 = 0
    cv2.destroyAllWindows()

    if dashboard_idx == 1:
        print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_idx], "状态正常,", status_list[dashboard_idx])
    else:
        area_danger[3][0] = 1
        threading.Thread(target=play_mp3, args=(area_mp3[zhuixingtong_idx],number_mp3[number_idx],status_mp3[dashboard_idx])).start()
        print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_idx], "状态异常,", status_list[dashboard_idx])
    
    if is_play_dashboard8 == 0 and area_danger[3][1] != 1: #########dashboard8
        threading.Thread(target=play_mp3, args=(area_mp3[zhuixingtong_idx],number_mp3[number_8],status_mp3[area_danger[3][1]])).start()
        print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_8], "状态异常,", status_list[area_danger[3][1]])




    robot.send_data(0x21010D05, 0, 0)
    time.sleep(0.1)
    robot.send_data(0x21010C03, 0, 0)
    time.sleep(0.1)       
    publish_cmd_vel(pub,1,0,0,1) #turn left
    dashboard_idx = detect_dashboard() ######dashboard 6
    number_idx = detect_number()
    zhuixingtong_idx = area_color[2]
    cv2.destroyAllWindows()
    if dashboard_idx == 1:
        print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_idx], "状态正常,", status_list[dashboard_idx])
    else:
        area_danger[2][1] = 1
        threading.Thread(target=play_mp3, args=(area_mp3[zhuixingtong_idx],number_mp3[number_idx],status_mp3[dashboard_idx])).start()
        print(zhuixingtong_list[zhuixingtong_idx], ",", number_list[number_idx], "状态异常,", status_list[dashboard_idx])
    
    robot.send_data(0x21010D05, 0, 0)
    time.sleep(0.1)
    robot.send_data(0x21010C03, 0, 0)
    time.sleep(0.1)      
    publish_cmd_vel(pub,3.2,0,0,1) #turn left
    time.sleep(0.1)    #ball4
    ball_color[3] = detect_ball()
    danger_idx = list(set(np.where(area_danger != np.full((4,2),-1))[0]))
    print('danger_idx',danger_idx)
    print('area_danger',area_danger)
    print('area_color',area_color)
    print('ball_color',ball_color)
    
    
    move_to_ball_center(False,pub) #ball4
    time.sleep(0.1)
    publish_cmd_vel(pub,0.5,0.5,0,0) #forward
    time.sleep(0.1)
    publish_cmd_vel(pub,0.5,-0.5,0,0) #backward
    time.sleep(0.1)
    
    publish_cmd_vel(pub,1.8,0,0,1) #turn left
    time.sleep(0.1)
    align_with_white_line(robot, pub, pipeline) #xunji
    time.sleep(0.1)
    publish_cmd_vel(pub,7.1,0.5,0,0) #forward
    time.sleep(0.1)
    publish_cmd_vel(pub,0.6,0,0,-1) #turn right
    move_to_ball_center(False,pub) #ball1
    time.sleep(0.1)
    publish_cmd_vel(pub,2.5,0.5,0,0) #forward
    time.sleep(0.1)
    publish_cmd_vel(pub,2.5,-0.5,0,0) #backward
    time.sleep(0.1)
    
    publish_cmd_vel(pub,1.8,0,0,1) #turn left
    time.sleep(0.1)
    align_with_white_line(robot, pub, pipeline) #xunji
    time.sleep(0.1)
    publish_cmd_vel(pub,6.9,0.5,0,0) #forward
    time.sleep(0.1)
    publish_cmd_vel(pub,0.6,0,0,-1) #turn right
    time.sleep(0.1)
    move_to_ball_center(False,pub) #ball2
    time.sleep(0.1)
    publish_cmd_vel(pub,2.5,0.5,0,0) #forward
    time.sleep(0.1)
    publish_cmd_vel(pub,2.5,-0.5,0,0) #backward
    time.sleep(0.1)


    publish_cmd_vel(pub,1.8,0,0,1) #turn left
    time.sleep(0.1)
    align_with_white_line(robot, pub, pipeline) #xunji
    time.sleep(0.1)
    publish_cmd_vel(pub,6.9,0.5,0,0) #forward
    time.sleep(0.1)
    publish_cmd_vel(pub,0.6,0,0,-1) #turn right
    time.sleep(0.1)
    move_to_ball_center(False,pub) #ball3
    time.sleep(0.1)
    publish_cmd_vel(pub,0.5,0.5,0,0) #forward
    time.sleep(0.1)
    publish_cmd_vel(pub,0.5,-0.5,0,0) #backward
    time.sleep(0.1)
    robot.stand_up()
    print("time_all:", time.time() - begin_time)
