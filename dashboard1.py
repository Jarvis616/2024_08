import time
import struct
import cv2 as cv
import cv2
import numpy as np
import threading
import pyrealsense2 as rs

from DashboardRecognition import DashboardRecognition
from Controller import Controller

Develop_Mode = True # True means use computer camera. False means use dog camera
client_address = ("192.168.1.103", 43897)
server_address = ("192.168.1.120", 43893)


global frame
global number
global is_update_number

if __name__ == '__main__':
    dashboard_detector = DashboardRecognition()
    controller = Controller(server_address)
    dashboard_counts = {}  # 记录仪表识别次数的字典
    start_time_1 = 0
    status = None
    depth = 0
    if Develop_Mode:
        # 配置 RealSense 摄像头
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # 开启 RealSense 摄像头
        pipeline.start(config)
        #cap = cv.VideoCapture(0)
    else:
        cap = cv.VideoCapture("/dev/video0", cv.CAP_V4L2)
        # cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
        # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

        stop_heartbeat = False
        # start to exchange heartbeat pack
        def heart_exchange(con):
            pack = struct.pack('<3i', 0x21040001, 0, 0)
            while True:
                if stop_heartbeat:
                    return
                con.send(pack)
                time.sleep(0.25)  # 4Hz

        heart_exchange_thread = threading.Thread(target=heart_exchange, args=(controller,))
        heart_exchange_thread.start()

        # stand up
        print("Wait 10 seconds and stand up......")
        pack = struct.pack('<3i', 0x21010202, 0, 0)
        controller.send(pack)
        time.sleep(5)
        controller.send(pack)
        time.sleep(5)
        controller.send(pack)
        print("Dog should stand up, otherwise press 'ctrl + c' and re-run the demo")

    # try to use CUDA
    if cv.cuda.getCudaEnabledDeviceCount() != 0:
        backend = cv.dnn.DNN_BACKEND_CUDA
        target = cv.dnn.DNN_TARGET_CUDA
    else:
        backend = cv.dnn.DNN_BACKEND_DEFAULT
        target = cv.dnn.DNN_TARGET_CPU
        print('CUDA is not set, will fall back to CPU.')

    status_buffer = [None] * 3

    while True:
        #frame = cv.imread('test1.png')
        #ret, frame = cap.read()
        # 获取 RealSense 摄像头的帧
        frames = pipeline.wait_for_frames()
        #if not ret:
        #    continue
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        fig_width = depth_frame.get_width()
        fig_height = depth_frame.get_height()
        
        image = np.asanyarray(color_frame.get_data())
        
        # 进行白平衡处理（示例：简单的灰度世界法）
        balanced_image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
        avg_a = np.average(balanced_image[:, :, 1])
        avg_b = np.average(balanced_image[:, :, 2])
        balanced_image[:, :, 1] = balanced_image[:, :, 1] - ((avg_a - 128) * (balanced_image[:, :, 0] / 255.0) * 1.1)
        balanced_image[:, :, 2] = balanced_image[:, :, 2] - ((avg_b - 128) * (balanced_image[:, :, 0] / 255.0) * 1.1)
        image = cv.cvtColor(balanced_image, cv.COLOR_LAB2BGR)
        
        # 转换为RGB
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        bbox = dashboard_detector.detect(image)
        
        if dashboard_detector.biggest is not None:
            depth = depth_frame.get_distance(dashboard_detector.biggest[0] , dashboard_detector.biggest[1] )
            #print(depth)
        cv2.putText(image, f"Distance: {depth:.2f} m", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        frame = dashboard_detector.visualize(image, status)
        temp_status = dashboard_detector.get_status(frame)
        
        if start_time_1 == 0:
            start_time_1 = time.time()
        # 更新手势识别次数
        if temp_status is not None:
            if temp_status in dashboard_counts:
                dashboard_counts[temp_status] += 1
            else:
                dashboard_counts[temp_status] = 1
        if time.time() - start_time_1 >= 3:
            # 找到最常识别的手势
            if len(dashboard_counts):
                status = max(dashboard_counts, key=dashboard_counts.get)
                start_time_1 = 0
                dashboard_counts = {}
        
        status_buffer.insert(0, status)
        status_buffer.pop()
        if status is not None and all(s == status_buffer[0] for s in status_buffer):
            print("当前仪表盘压力值为 {}".format(status))
        
        cv.imshow("Danger Sign Recognition", frame)
        k = cv.waitKey(1)
        if k == 113 or k == 81:  # q or Q to quit
            print("Demo is quiting......")
            if not Develop_Mode:
                controller.drive_dog("squat")
            cap.release()
            cv.destroyWindow("Danger Sign Recognition")
            stop_heartbeat = True
            is_update_number = False
            break
