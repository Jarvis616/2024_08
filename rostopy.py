import rospy
import time
from geometry_msgs.msg import Twist
from move_lite3 import HeartBeat, ControlRobot




def publisher_init():
    rospy.init_node('cmd_vel_publisher', anonymous=True)
    
    # 创建一个 Publisher，发布到 /cmd_vel 话题，消息类型为 geometry_msgs/Twist
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    return pub


def publish_cmd_vel(pub,t,velocity_x,velocity_y,anguler_z): # time 为运行时间； x 为前向速度； y 为侧向速度； z 为角速度
    # 设置发布频率
    rate = rospy.Rate(10) # 10Hz
    start_time = time.time()
    # 创建一个 Twist 消息
    twist = Twist()
    while (time.time() < start_time + t) :

        
        # 设置线速度
        twist.linear.x = velocity_x  # 线速度 x 方向
        twist.linear.y = velocity_y  # 线速度 y 方向
        twist.linear.z = 0.0  # 线速度 z 方向
        
        # 设置角速度
        twist.angular.x = 0.0  # 角速度 x 方向
        twist.angular.y = 0.0  # 角速度 y 方向
        twist.angular.z = anguler_z # 角速度 z 方向
        
        # 发布消息
        pub.publish(twist)
        # 打印已发布的信息（可选）
        #rospy.loginfo("Published Twist message: linear x = %.2f, linear y = %.2f, angular z = %.2f", twist.linear.x, twist.linear.y, twist.angular.z)
        # 休眠，保持发布频率
        rate.sleep()
    
    if t == -1: 
        # 设置线速度
        twist.linear.x = velocity_x  # 线速度 x 方向

        twist.linear.y = velocity_y  # 线速度 y 方向
        twist.linear.z = 0.0  # 线速度 z 方向
    
        # 设置角速度
        twist.angular.x = 0.0  # 角速度 x 方向
        twist.angular.y = 0.0  # 角速度 y 方向
        twist.angular.z = anguler_z  # 角速度 z 方向  
    else:
        # 设置线速度
        twist.linear.x = 0.0  # 线速度 x 方向
        twist.linear.y = 0.0  # 线速度 y 方向
        twist.linear.z = 0.0  # 线速度 z 方向
    
        # 设置角速度
        twist.angular.x = 0.0  # 角速度 x 方向
        twist.angular.y = 0.0  # 角速度 y 方向
        twist.angular.z = 0.0  # 角速度 z 方向
    # 发布消息
    pub.publish(twist)
    # 打印已发布的信息（可选）
    #rospy.loginfo("Published Twist message: linear x = %.2f, linear y = %.2f, angular z = %.2f", twist.linear.x, twist.linear.y, twist.angular.z)
    # 休眠，保持发布频率
    rate.sleep()
    
    


'''
if __name__ == '__main__':
    h = HeartBeat()
    time.sleep(1)  # waiting for heart beat

    robot = ControlRobot()
    robot.send_data(0x21010C03, 0, 0)
    time.sleep(0.5)
    pub = publisher_init()
    try:
        publish_cmd_vel(pub,10,1,0,0)

    except rospy.ROSInterruptException:
        pass
'''
