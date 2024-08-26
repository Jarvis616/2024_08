import pyrealsense2 as rs
import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from move_lite3 import HeartBeat, ControlRobot
import time
from rostopy import publish_cmd_vel, publisher_init
def calculate_length(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

class PIDController:
    '''
    def __init__(self, kp = 0.005, ki = 0, kd = 0.001):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.error_1 = 0
        self.error_2 = 0
        self.last_out = 0
    '''
    def __init__(self, kp=0.005, ki=0.0, kd=0.001):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
    def compute(self,error):
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        pid_out = self.kp * error + self.ki * self.integral + self.kd * derivative
        if pid_out > 0.1:
            pid_out = 0.1
        elif pid_out < -0.1:
            pid_out = -0.1
        #print(pid_out)
        return pid_out
    '''
    def compute(self, error):
        pid_out = self.last_out + self.kp * (error - self.error_1) + self.ki * error + self.kd * (error - 2*self.error_1 + self.error_2)

        if pid_out > 0.1:
            pid_out = 0.1
        elif pid_out < -0.1:
            pid_out = -0.1
        
        self.error_2 = self.error_1
        self.error_1 = error
        self.last_out = pid_out

        return pid_out
    '''

def move_ball_to_position(distance_x, distance_y):


    # PID controllers for linear and angular velocities
    pid_linear_x = PIDController()
    pid_linear_y = PIDController()


    # 计算需要的线速度和角速度
    linear_speed_x = pid_linear_x.compute(distance_x)
    linear_speed_y = pid_linear_y.compute(distance_y)

    return linear_speed_x, linear_speed_y
'''
def detect_angle(frame):
    angle = None
    # 转换为HSV颜色空间
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
    # 设置阈值提取白色区域
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask = cv2.inRange(hsv_image, lower_white, upper_white)
        
    # 使用Canny边缘检测
    edges = cv2.Canny(mask, 50, 150)
        
    # 使用霍夫变换检测线条
    #lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    lines = cv2.HoughLinesP(edges, 0.5, np.pi / 360, threshold=30, minLineLength=30, maxLineGap=20)
        
    if lines is not None:
        lines_with_lengths = []
            
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = calculate_length(x1, y1, x2, y2)
            lines_with_lengths.append((length, line[0]))
                
        # 按长度排序
        lines_with_lengths.sort(reverse=True, key=lambda x: x[0])
                
        if len(lines_with_lengths) >= 1:
            # 第一长的线
            longest_line = lines_with_lengths[0][1]
            x1, y1, x2, y2 = longest_line
            #cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            length1 = lines_with_lengths[0][0]       
            angle1 = np.degrees(np.arctan2(y2-y1, x2-x1))
            if angle1 < 0:
                angle1 = angle1 + 180
            if len(lines_with_lengths) >= 2:
                
                # 第二长的线
                second_longest_line = lines_with_lengths[1][1]
                length2 = lines_with_lengths[1][0]
                print(length1, length2)
                x1, y1, x2, y2 = second_longest_line
                #cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
                angle2 = np.degrees(np.arctan2(y2-y1, x2-x1))
                if angle2 < 0:
                    angle2 = angle2 + 180
                if length1 - length2 < 50:
                    angle = (angle2 + angle1) / 2
                else:
                    angle = angle1
            else:
                angle = angle1
            print(f"angle: {angle}")
    else:
        angle = None
    
    return angle
'''
def detect_cx(frame):
    # 将图像转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 使用二值化检测白线
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    # 找到白线的轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        # 计算最小外接矩形
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        vertex1,vertex2,vertex3,vertex4 = box
        if np.linalg.norm(vertex1-vertex2) > np.linalg.norm(vertex2-vertex3):
            angle1 = np.degrees(np.arctan2(vertex1[1]-vertex2[1], vertex1[0]-vertex2[0]))
            cv2.line(frame, tuple(vertex1), tuple(vertex2), (0, 255, 0), 2)
            if angle1 < 0:
                angle1 = angle1 + 180
            angle2 = np.degrees(np.arctan2(vertex3[1]-vertex4[1], vertex3[0]-vertex4[0]))
            if angle2 < 0:
                angle2 = angle2 + 180
            angle = (angle1 + angle2)/2
        else:
            angle1 = np.degrees(np.arctan2(vertex2[1]-vertex3[1], vertex2[0]-vertex3[0]))
            cv2.line(frame, tuple(vertex3), tuple(vertex2), (0, 255, 0), 2)
            if angle1 < 0:
                angle1 = angle1 + 180
            angle2 = np.degrees(np.arctan2(vertex1[1]-vertex4[1], vertex1[0]-vertex4[0]))
            if angle2 < 0:
                angle2 = angle2 + 180
            angle = (angle1 + angle2)/2
            
        cx = int(np.mean(box[:, 0]))

    else:
        cx = None
        angle = None
    
    return angle,cx

def align_with_white_line(robot, pub, pipeline, center_cx = 150, center_angle = 112, left_or_right=0.1):
    rotate_flag = 1
    angle = 10000
    cx = 10000
    t0 = time.time()
    # PID controllers for angular and linear velocities
    pid_angular = PIDController()
    pid_linear = PIDController()



    try:
        while not rospy.is_shutdown():
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            

            if not color_frame:
                continue

            # Convert images to numpy arrays
            frame = np.asanyarray(color_frame.get_data())
            height, width = frame.shape[:2]
            frame = frame[int(height//4*3):, :]

            # 检测白线的位置
            #angle = detect_angle(frame)
            angle,cx = detect_cx(frame)
            if angle is not None and cx is not None:
                if abs(angle - center_angle) <= 3 and abs(cx - center_cx) <= 10:
                    print("cx", cx)
                    print('angle',angle)
                    break
                # 计算角速度
                angular_speed = pid_angular.compute(center_angle - angle)  #112
                # 计算线速度
                linear_speed = pid_linear.compute(center_cx - cx)    #150
                
      
                publish_cmd_vel(pub,-1,0,linear_speed,angular_speed)

                # 显示图像和白线位置
                #box = cv2.boxPoints(cv2.minAreaRect(max(cv2.findContours(cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY)[1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)))
                #box = np.int0(box)
                #cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
                cv2.circle(frame, (cx, frame.shape[0] // 2), 5, (0, 255, 0), -1)
                cv2.imshow("White Line Detection", frame)
            else:
                publish_cmd_vel(pub,-1,0,left_or_right,0)
                continue
                if rotate_flag % 2 == 1:
                    angular_z = 0.1 * rotate_flag
                    rotate_flag = rotate_flag + 3
                    publish_cmd_vel(pub,-1,0,0,angular_z)
                    cv2.imshow("White Line Detection", frame)
                else:
                    angular_z = -0.1 * rotate_flag
                    rotate_flag = rotate_flag + 3
                    publish_cmd_vel(pub,-1,0,0,angular_z)
                    cv2.imshow("White Line Detection", frame)
		
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        
    finally:
        print("line_angle_Done",time.time()-t0)
        # publish_cmd_vel(pub,-1,0,0,0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # Initialize RealSense camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    h = HeartBeat()
    pub = publisher_init() #init pub
    time.sleep(1)  # waiting for heart beat

    robot = ControlRobot()
    robot.stand_up()
    time.sleep(2)
    
    robot.send_data(0x21010C03, 0, 0)
    time.sleep(0.5)
    try:
        align_with_white_line(robot, pub, pipeline, 497, 65) 
    except rospy.ROSInterruptException:
        pass
    #publish_cmd_vel(pub,2,0.5,0,0)
    time.sleep(3)
    robot.stand_up()

