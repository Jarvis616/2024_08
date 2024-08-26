import mediapipe as mp
import cv2
import time
import pyrealsense2 as rs
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from collections import deque, Counter
font_path = './YZGGPB.ttf'
font = ImageFont.truetype(font_path,50)

def get_angle(v1,v2):
    angle = np.dot(v1,v2)/(np.sqrt(np.sum(v1*v1))*np.sqrt(np.sum(v2*v2)))
    angle = np.arccos(angle)/3.14*180
    
    return angle 

def gesture_move(up_fingers,list_lms):
    #print(((list_lms[4].x-list_lms[8].x)**2+(list_lms[4].y-list_lms[8].y)**2+(list_lms[4].z-list_lms[8].z)**2) / ((list_lms[8].x-list_lms[20].x)**2+(list_lms[8].y-list_lms[20].y)**2+(list_lms[8].z-list_lms[20].z)**2))
    
    if len(up_fingers) == 5 and ((list_lms[4].x-list_lms[8].x)**2+(list_lms[4].y-list_lms[8].y)**2+(list_lms[4].z-list_lms[8].z)**2) / ((list_lms[12].x-list_lms[16].x)**2+(list_lms[12].y-list_lms[16].y)**2+(list_lms[12].z-list_lms[16].z)**2) > 0.9:
        move = 1
    elif len(up_fingers) == 2 and up_fingers[0] == 4 and up_fingers[1] == 8 :
        move = 0
    elif len(up_fingers) == 3 and up_fingers[0] == 4 and up_fingers[1] == 8 and ((list_lms[4].x-list_lms[8].x)**2+(list_lms[4].y-list_lms[8].y)**2+(list_lms[4].z-list_lms[8].z)**2) / ((list_lms[8].x-list_lms[20].x)**2+(list_lms[8].y-list_lms[20].y)**2+(list_lms[8].z-list_lms[20].z)**2) < 1:
        move = 0
    elif len(up_fingers) == 2 and up_fingers[0] == 4 and up_fingers[1] == 20:
        move = 5
    elif len(up_fingers) == 2 and up_fingers[0] == 8 and up_fingers[1] == 12:
        move = 3
    elif len(up_fingers) == 3 and up_fingers[0] == 8 and up_fingers[1] == 12 and (list_lms[4].x-list_lms[12].x)**2+(list_lms[4].y-list_lms[12].y)**2+(list_lms[4].z-list_lms[12].z)**2 > 0.003:
        move = 3
    elif len(up_fingers) == 3 and up_fingers[0] == 4 and up_fingers[1] == 8 and up_fingers[2] == 20:
        move = 2
    elif len(up_fingers) == 3 and up_fingers[0] == 8 and up_fingers[1] == 16 and up_fingers[2] == 20:
        move = 6
    elif len(up_fingers) == 3 and up_fingers[0] == 12 and up_fingers[1] == 16 and up_fingers[2] == 20: #and (list_lms[4].x-list_lms[8].x)**2+(list_lms[4].y-list_lms[8].y)**2+(list_lms[4].z-list_lms[8].z)**2 < 0.03:
        move = 4
    elif len(up_fingers) == 5 and up_fingers[0] == 4 and ((list_lms[4].x-list_lms[8].x)**2+(list_lms[4].y-list_lms[8].y)**2+(list_lms[4].z-list_lms[8].z)**2) / ((list_lms[12].x-list_lms[16].x)**2+(list_lms[12].y-list_lms[16].y)**2+(list_lms[12].z-list_lms[16].z)**2) < 0.9:
        move = 4
    else:
        move = 7
    return move
    

def get_circle_points(center, radius):
    #获取以 center 为中心、radius 为半径的圆上的所有点的坐标
    circle_points = []
    x0, y0 = center
    for theta in np.linspace(0, 2*np.pi / (radius + 1), 100):
        x = x0 + radius * np.cos(theta)
        y = y0 + radius * np.sin(theta)
        circle_points.append((x, y))
    return circle_points


if __name__ == "__main__":
    # is_get_gesture = 1
    # start_time_1 = 0
    # start_time_2 = 0
    # start_time_3 = 0
    gesture_list = ["前进", "后退", "原地扭身", "左平移", "右平移", "左旋转", "右旋转", " "]
    str_guester = " "
    gesture_counts = deque(maxlen=30)  # 记录手势识别次数的字典
    # 配置 RealSense 摄像头
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # 开启 RealSense 摄像头
    pipeline.start(config)
    # cap = cv2.VideoCapture(0)
    # 定义手 检测对象
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils
  
    while True:


        # 获取 RealSense 摄像头的帧
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        fig_width = depth_frame.get_width()
        fig_height = depth_frame.get_height()

        # 将帧转换为 OpenCV 格式
        img = np.asanyarray(color_frame.get_data())
     
        # # 读取一帧图像
        # success, img = cap.read()
        # if not success:
        #     continue
        image_height, image_width, _ = np.shape(img)
        
        # 转换为RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 得到检测结果
        results = hands.process(imgRGB)
        
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            index_finger_landmark = hand.landmark[mpHands.HandLandmark.WRIST]
            index_finger_landmark_1 = hand.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP]
            index_x = (index_finger_landmark.x + index_finger_landmark_1.x)/2
            index_y = (index_finger_landmark.y + index_finger_landmark_1.y)/2
            depth = 0
            i = 0
            while depth == 0:
                circle_points = get_circle_points((index_x, index_y), i)
                for point in circle_points:
                    x, y = point
                    index_finger_x = int(x * color_frame.width)
                    index_finger_y = int(y * color_frame.height)
                    if index_finger_x > fig_width - 1:
                        index_finger_x = fig_width - 1
                    if index_finger_x < 0:
                        index_finger_x = 0
                    if index_finger_y > fig_height - 1:
                        index_finger_y = fig_height - 1
                    if index_finger_y < 0:
                        index_finger_y = 0
                    depth = depth_frame.get_distance(index_finger_x, index_finger_y)
                    if depth:
                        break
                i = i + 0.05
                if i > 300:
                    break
            distance = depth
            if distance is None:
                distance = 0
            cv2.circle(img, (index_finger_x, index_finger_y), 10, (0, 255, 0), 1)
            # 在图像上显示距离信息
            cv2.putText(img, f"Distance: {distance:.2f} m", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            mpDraw.draw_landmarks(img,hand,mpHands.HAND_CONNECTIONS)
            
            # 采集所有关键点的坐标
            list_lms = []    
            for i in range(21):
                pos_x = hand.landmark[i].x*image_width
                pos_y = hand.landmark[i].y*image_height
                list_lms.append([int(pos_x),int(pos_y)])
            
            # 构造凸包点
            list_lms = np.array(list_lms,dtype=np.int32)
            hull_index = [0,1,2,3,6,10,14,19,18,17,0]
            hull = cv2.convexHull(list_lms[hull_index,:])
            # 绘制凸包
            cv2.polylines(img,[hull], True, (0, 255, 0), 2)


            n_fig = -1
            ll = [4,8,12,16,20] 
            up_fingers = []
                
            for i in ll:
                pt = (int(list_lms[i][0]),int(list_lms[i][1]))
                dist= cv2.pointPolygonTest(hull,pt,True)
                if dist <0:
                    up_fingers.append(i)
                
            # print(up_fingers)
            # print(list_lms)
            # print(np.shape(list_lms))
            #str_guester = gesture_move(up_fingers,list_lms)
            gesture = gesture_move(up_fingers,hand.landmark)
            gesture_counts.append(gesture)
            
            if len(gesture_counts) > 0:
                count = Counter(gesture_counts)
                most_common_num, most_common_count = count.most_common(1)[0]
                str_guester = gesture_list[most_common_num]
            

            #print((hand.landmark[4].x-hand.landmark[8].x)**2+(hand.landmark[4].y-hand.landmark[8].y)**2+(hand.landmark[4].z-hand.landmark[8].z)**2)
            
            img_pil = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            draw.text((150,50),str_guester,font = font,fill = (255,0,0))
            # while True:
            #     start_time = time.time()  # 记录开始时间
            #     draw.text((150,50),str_guester,font = font,fill = (255,0,0))
            #     if time.time() - start_time >= 6: #假设机器人动作需要6s
            #         break
            img = cv2.cvtColor(np.array(img_pil),cv2.COLOR_RGB2BGR)
            
            
            #cv2.putText(img,' %s'%(str_guester),(90,90),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),4,cv2.LINE_AA)
            
                
             
            for i in ll:
                pos_x = hand.landmark[i].x*image_width
                pos_y = hand.landmark[i].y*image_height
                # 画点
                cv2.circle(img, (int(pos_x),int(pos_y)), 3, (0,255,255),-1)

        
            
        cv2.imshow("hands",img)

        key =  cv2.waitKey(1) & 0xFF   

        # 按键 "q" 退出
        if key ==  ord('q'):
            break
    pipeline.stop()
    cv2.destroyAllWindows() 
