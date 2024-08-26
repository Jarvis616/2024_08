import socket, struct, time, threading

class HeartBeat():
    def __init__(self):
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.send_addr = ('192.168.1.120', 43893)
        self.heart_task = threading.Thread(target=self.send_heartbeat)
        self.heart_task.start()

    def send_heartbeat(self):
        while True:
            data = struct.pack("<3i", 0x21040001, 0, 0)
            self.udp_socket.sendto(data, self.send_addr)
            time.sleep(0.5)  # sending command frequency not lower than 2HZ

class ControlRobot():
    def __init__(self):
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.send_addr = ('192.168.1.120', 43893)

    def send_data(self, code, value, type):
        data = struct.pack("<3i", code, value, type)
        self.udp_socket.sendto(data, self.send_addr)

    def stand_up(self):
        self.send_data(0x21010202, 0, 0)
        self.send_data(0x21010D06, 0, 0)
        time.sleep(2)

    def forward(self, duration):
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_data(0x21010130, 13000, 0)
            time.sleep(0.05)  # sending command frequency not lower than 20HZ

    def backward(self, duration):
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_data(0x21010130, -13000, 0)
            time.sleep(0.05)  # sending command frequency not lower than 20HZ

    def twist(self, duration):
        start_time = time.time()
        self.send_data(0x21010204, 0, 0)
        time.sleep(0.05)  # sending command frequency not lower than 20HZ
        while time.time() - start_time < duration:
            continue

            

    def move_left(self, duration):
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_data(0x21010131, -20000, 0)
            time.sleep(0.05)

    def move_right(self, duration):
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_data(0x21010131, 20000, 0)
            time.sleep(0.05)

    def turn_left(self):
        self.send_data(0x21010C0A, 13, 0)
        time.sleep(2)
        self.send_data(0x21010C06, 2, 0)

    def turn_right(self):
        self.send_data(0x21010C0A, 14, 0)
        time.sleep(2)

    def turn_left1(self, duration):
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_data(0x21010135, -13000, 0)
            time.sleep(0.05)  

    def turn_right1(self, duration):
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_data(0x21010135, 13000, 0)
            time.sleep(0.05)    

    def stop(self):
        self.send_data(0x21020C0E, 20000, 0)
        time.sleep(0.05)
        
    def turn_to_Auto(self):
    	self.send_data(0x21010C03,0,0)
    	time.sleep(0.05)
    	
    def turn_to_Man(self):
    	self.send_data(0x21010C02,0,0)
    	time.sleep(0.05)



if __name__ == '__main__':
    h = HeartBeat()
    time.sleep(1)  # waiting for heart beat

    robot = ControlRobot()
    
    robot.stand_up()
    print("stand_up")
    time.sleep(3)  # waiting for standing up
    
    robot.turn_to_Man()
    print("Auto")
    time.sleep(0.05)
    
#    robot.turn_to_Man()
#    print("Auto")
#    time.sleep(0.05)  
    
#     robot.forward(3)  # walking forward for 3s
#     print("forward")
#     time.sleep(1)

#     robot.voice_tl_90()  # turning left 90°
#     print("turn_left_90")
#     time.sleep(3)

#     robot.move_left(4)  # moving left for 4s
#     print("move_left")
#     time.sleep(1)
