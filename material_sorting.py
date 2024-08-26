import argparse
import time
from pathlib import Path

import cv2
import torch
import math
import os
import threading

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


from angle_cx import align_with_white_line, move_ball_to_position
import time
from rostopy import publisher_init, publish_cmd_vel
import signal
from hiwonder_interfaces.msg import SerialServoMove

font_path = './YZGGPB.ttf'
font = ImageFont.truetype(font_path,50)

# 总线舵机数据检测
bus_servo_data_detection = False
running = True

def shutdown(signum, frame):
    global running
    running = False
    rospy.loginfo('shutdown')
    rospy.signal_shutdown('shutdown')
    
# 总线舵机数据回调函数
def bus_servo_data_callback(msg):
    global bus_servo_data_detection
    print(msg)
    if msg.servo_id != []: #判断该话题的ID是否为空
        bus_servo_data_detection = True

def bus_servo_controls(id=0,
                       position=0,
                       duration=0.0):
                       
    #bus_servo_data =[]
    # 设置总线舵机消息类型
    data = SerialServoMove()
    data.servo_id = id #总线舵机ID      
    data.position = position #总线舵机角度[0-1000]
    data.duration = duration #总线舵机运行时间
    bus_servo_pub.publish(data) #发布数据

def arm_zhuaqu():
    start_time = time.time()
    while time.time() - start_time < 2:
        bus_servo_controls(id = 10,position = 750,duration=500) #发布数据
        rospy.sleep(0.25)
    
    
    start_time = time.time()
    while time.time() - start_time < 2:
        bus_servo_controls(id = 3,position = 850,duration=500) #发布数据
        rospy.sleep(0.25)
    
    start_time = time.time()
    while time.time() - start_time < 2:
        bus_servo_controls(id = 4,position = 600,duration=500) #发布数据
        rospy.sleep(0.25)
    



def arm_zhengdui():

    start_time = time.time()
    while time.time() - start_time < 2:
        bus_servo_controls(id = 4,position = 1000,duration=500) #发布数据
        rospy.sleep(0.25)
    
    start_time = time.time()
    while time.time() - start_time < 2:
        bus_servo_controls(id = 3,position = 350,duration=500) #发布数据
        rospy.sleep(0.25)
    

    start_time = time.time()
    while time.time() - start_time < 2:
        bus_servo_controls(id = 4,position = 750,duration=500) #发布数据
        rospy.sleep(0.25)

    start_time = time.time()
    while time.time() - start_time < 2:
        bus_servo_controls(id = 10,position = 0,duration=500) #发布数据
        rospy.sleep(0.25)

def arm_fangxia():

    start_time = time.time()
    while time.time() - start_time < 2:
        bus_servo_controls(id = 4,position = 1000,duration=500) #发布数据
        rospy.sleep(0.25)
    
    start_time = time.time()
    while time.time() - start_time < 2:
        bus_servo_controls(id = 3,position = 400,duration=500) #发布数据
        rospy.sleep(0.25)
    

    start_time = time.time()
    while time.time() - start_time < 2:
        bus_servo_controls(id = 4,position = 750,duration=500) #发布数据
        rospy.sleep(0.25)

    start_time = time.time()
    while time.time() - start_time < 2:
        bus_servo_controls(id = 10,position = 0,duration=500) #发布数据
        rospy.sleep(0.25)

def arm_init():
    start_time = time.time()
    while time.time() - start_time < 2:
        bus_servo_controls(id = 1,position = 500,duration=500) #发布数据
        rospy.sleep(0.25)
    #start_time = time.time()
    #while time.time() - start_time < 2:
    #    bus_servo_controls(id = 2,position = 100,duration=500) #发布数据
    #    rospy.sleep(0.25)
    
    start_time = time.time()
    while time.time() - start_time < 2:
        bus_servo_controls(id = 3,position = 850,duration=500) #发布数据
        rospy.sleep(0.25)
    
    start_time = time.time()
    while time.time() - start_time < 2:
        bus_servo_controls(id = 4,position = 600,duration=500) #发布数据
        rospy.sleep(0.25)
        
    start_time = time.time()
    while time.time() - start_time < 2:
        bus_servo_controls(id = 5,position = 500,duration=500) #发布数据
        rospy.sleep(0.25)
    start_time = time.time()
    while time.time() - start_time < 2:
        bus_servo_controls(id = 10,position = 0,duration=500) #发布数据
        rospy.sleep(0.25)

def move_to_YZT_center(save_img,pub):
    distance = 10000
    distance_x = 10000
    distance_x_1 = 10000
    distance_y = 10000
    dashboard_target_x = 406
    dashboard_target_y = 358
    dashboard_target_x2_x1 = 145

    
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
            # 如果距离足够小，认为到达目标点，停止运动
            if distance_x_1 <= 5 and abs(distance_y) <= 8:
                publish_cmd_vel(pub, -1, 0, 0, 0)
                break
            if time.time() - t0 > 100:
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
                            #distance_x = dashboard_target_y - center_dashboard_y
                            distance_x = dashboard_target_x2_x1 - (x2 - x1)
                            distance_x_1 = dashboard_target_x2_x1 - 3 - (x2 - x1)
                            distance = math.sqrt(distance_x**2 + distance_y**2)
                            print("ball_center",center_dashboard_x,center_dashboard_y)
                            print("x2-x1",x2-x1)
                            move_x, move_y = move_ball_to_position(distance_x, distance_y)

                    
            # Show results
            cv2.imshow('RealSense', img0)
            cv2.waitKey(5)
            # 发布消息
            
            if move_x is not None and move_y is not None:
                publish_cmd_vel(pub, -1, move_x, move_y, 0)
            else:
                publish_cmd_vel(pub, -1 , -0.01, -0.1, 0) 
            

    finally:
        print(f'Ball_Done. ({time.time() - t0:.3f}s)')
        cv2.destroyAllWindows()
        
if __name__ == '__main__':

    # Initialize RealSense camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device('147322073278')
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    
    
    pub = publisher_init() #init pub
    # rospy.init_node('bus_servo_demo', anonymous=True) #初始化节点
    signal.signal(signal.SIGINT, shutdown)
    rospy.wait_for_service('/jetarm_sdk/get_loggers')
    #发布总线舵机话题
    bus_servo_pub = rospy.Publisher('/jetarm_sdk/serial_servo/move', SerialServoMove, queue_size=1)
    #接收总线舵机话题
    bus_servo_sub = rospy.Subscriber('/jetarm_sdk/serial_servo/move', SerialServoMove, bus_servo_data_callback)

    #find_dashboard/pointer/sign_model
    parser2 = argparse.ArgumentParser()
    parser2.add_argument('--weights', nargs='+', type=str, default='yolo_models/best_yzt.pt', help='model.pt path(s)')
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

    robot = ControlRobot()

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
            cv2.destroyAllWindows()
            break    

    
    robot.send_data(0x21010C02, 0, 0) #shou dong mo shi
    time.sleep(0.05)

    begin_time = time.time()
    robot.stand_up()
    print("stand_up")
    time.sleep(0.5)  # waiting for standing up

    arm_init()

    robot.send_data(0x21010C03, 0, 0) #zi zhu mo shi
    time.sleep(0.05)

    publish_cmd_vel(pub,-1,0,0,0) 
    time.sleep(2)
    publish_cmd_vel(pub,0.5,0,-1,0) #move right
    time.sleep(0.5)
    publish_cmd_vel(pub,1.8,1,0,0) #forward
    time.sleep(0.5)
    move_to_YZT_center(False,pub)
    time.sleep(0.5)
    

    arm_zhengdui()
    arm_zhuaqu()
    
    publish_cmd_vel(pub,3.2,0,0.5,0)  #move left
    arm_fangxia()
    
    print("total_time", time.time() - begin_time)
    
    
    
