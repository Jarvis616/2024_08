import argparse
import time
from pathlib import Path

import cv2
import torch
import math
import torch.backends.cudnn as cudnn
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
import torch.nn as nn
from torchvision import transforms
from PIL import Image

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



def xyxy_in_box(xyxy_small, xyxy_large):
    """
    Check if xyxy_small is completely inside xyxy_large
    Args:
        xyxy_small (list): Coordinates of the smaller box in format [x1, y1, x2, y2]
        xyxy_large (list): Coordinates of the larger box in format [x1, y1, x2, y2]
    Returns:
        bool: True if xyxy_small is inside xyxy_large, False otherwise
    """
    
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


    theta1 = np.arctan2(pointer_y2- pointer_y1, pointer_x2 - pointer_x1)
    theta2 = np.arctan2(center_sign_y - center_dashboard_y, center_sign_x - center_dashboard_x)

    theta = theta1 - theta2
    theta = np.degrees(theta)
    if theta < 0:
        theta = theta + 360
    
    if theta < 127.5:
        print("low")
        cv2.putText(img0, "status: low", (30, 50), 0, 1.5, (225, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    elif theta >= 127.5 and theta < 236.5:
        print("mid")
        cv2.putText(img0, "status: mid", (30, 50), 0, 1.5, (225, 255, 255), thickness=2,  lineType=cv2.LINE_AA)
    elif theta >= 236.5 and theta < 360:
        print("high")
        cv2.putText(img0, "status: high", (30, 50), 0, 1.5, (225, 255, 255), thickness=2,  lineType=cv2.LINE_AA)

        
def detect(save_img=False):

    robot.stand_up()
    time.sleep(1)
    
    # Initialize RealSense camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # variable initialized here
    gesture_classified_result = deque(maxlen=10)

    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
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
                    model(img, augment=opt.augment)[0]

            # # Resize image
            # img = cv2.resize(img0, (imgsz, imgsz))

            # # Convert to Tensor and adjust channel order
            # img = img.transpose(2, 0, 1)  # HWC to CHW
            # img = np.ascontiguousarray(img)  # Ensure contiguous memory
            # img = torch.from_numpy(img).to(device)
            # img = img.half() if half else img.float()  # uint8 to fp16/32
            # img /= 255.0  # 0 - 255 to 0.0 - 1.0
            # img = img.unsqueeze(0)  # Add batch dimension

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=opt.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
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
                        if conf > 0.5:
                            #gesture_classified_result.append(int(cls))
                            label = f'{conf:.2f}'
                            # label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy[0], img0, label=label, color=colors[int(cls)], line_thickness=1)
                            #ball_center_x, ball_center_y = int((xyxy[0][0] + xyxy[0][2]) // 2), int((xyxy[0][1] + xyxy[0][3]) // 2)
                            #print(ball_center_x, ball_center_y)
                            cropped_img = img0[int(xyxy[0][1]):int(xyxy[0][3]),int(xyxy[0][0]):int(xyxy[0][2])]
                            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                            resized_img = cv2.resize(cropped_img,(28,28))
                            resized_img = transforms.ToTensor()(resized_img).unsqueeze(0)
                            output = lenet5_model(resized_img.to(device))
                            _,predicted_label = torch.max(output,1)
                            gesture_classified_result.append(predicted_label.item())
                            
            
            if len(gesture_classified_result) > 0:
                img_pil = Image.fromarray(cv2.cvtColor(img0,cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                count = Counter(gesture_classified_result) 
                most_common_num, most_common_count = count.most_common(1)[0] 
                if most_common_num == 0:
                    print("0")
                    #draw.text((150,50),"前进",font = font,fill = (255,0,0)) 
                elif most_common_num == 1:
                    print("1")
                    #draw.text((150,50),"后退",font = font,fill = (255,0,0)) 
                elif most_common_num == 2:
                    print("2")
                    #draw.text((150,50),"原地扭身",font = font,fill = (255,0,0)) 
                elif most_common_num == 3:
                    print("3")
                    #draw.text((150,50),"左平移",font = font,fill = (255,0,0))
                elif most_common_num == 4:
                    print("4")
                    #draw.text((150,50),"后退",font = font,fill = (255,0,0)) 
                elif most_common_num == 5:
                    print("5")
                    #draw.text((150,50),"原地扭身",font = font,fill = (255,0,0)) 
                elif most_common_num == 6:
                    print("6")
                    #draw.text((150,50),"左平移",font = font,fill = (255,0,0))
                elif most_common_num == 7:
                    print("7")
                    #draw.text((150,50),"后退",font = font,fill = (255,0,0)) 
                elif most_common_num == 8:
                    print("8")
                    #draw.text((150,50),"原地扭身",font = font,fill = (255,0,0)) 
 
                img0 = cv2.cvtColor(np.array(img_pil),cv2.COLOR_RGB2BGR)
            
            
            
            # Show results
            cv2.imshow('RealSense', img0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                 robot.stand_up()
                 time.sleep(3)
                 break
            
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    print(f'Done. ({time.time() - t0:.3f}s)')



if __name__ == '__main__':
    h = HeartBeat()
    time.sleep(1)  # waiting for heart beat

    robot = ControlRobot()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='number/best_new.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='datasets/images/test', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
