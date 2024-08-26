import cv2 as cv
import numpy as np
import pytesseract
import cv2

# 找到轮廓上的其他点
def find_points_on_line(contour, vx, vy, x0, y0, center_x, center_y):
    line_points = []  # 保存拟合直线上的轮廓点

    # 获取直线的单位向量
    unit_vector = np.squeeze(np.array([vx, vy]),axis = 1)
    #print(unit_vector)
    max_distance = 0  # 保存最远距离
    farthest_point = None  # 保存距离中心点最远的点
    #print(contour)
    for point in contour:
        # 将轮廓点转换为 numpy 数组
        #print(point)
        point = np.array(point[0])
        #print(point)
        

        # 计算直线上点到参考点的向量
        ref_point = np.squeeze(np.array([x0, y0]),axis = 1)
        #print(ref_point)
        vec_to_point = point - ref_point
        #print(vec_to_point)

        # 计算点到直线的距离（点到直线的投影长度）
        distance_to_line = (np.cross(vec_to_point, unit_vector))

        # 设置一个阈值，表示点在直线上的容忍范围
        threshold = 0.1

        # 如果点到直线的距离在阈值范围内，则认为点在直线上
        if distance_to_line < threshold:
            line_points.append(point)

            # 计算点到中心点的距离
            distance_to_center = np.linalg.norm(point - np.array([center_x, center_y]))

            # 更新最远距离和最远点
            if distance_to_center > max_distance:
                max_distance = distance_to_center
                farthest_point = point

    #print(farthest_point[0])
    if farthest_point is None:
        return None
    else:
        return int(farthest_point[0]),int(farthest_point[1])

class DashboardRecognition:
    def __init__(self, min_area=50 * 50):
        self.min_area = min_area
        self.image_width = 0
        self.image_height = 0
        self.bbox = None
        self.dashboard_status = None
        self.biggest = None



    def detect(self, image, bbox=None):
        if bbox is None:
            bias = np.array([0, 0]).astype(np.int32)  # hand landmarks bias to left-top
        else:
            bias = bbox[0]  # hand landmarks bias to left-top
            # crop image
            image = image[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0], :]
        self.image_height, self.image_width = image.shape[:2]
        self.bbox = None
        # 转换为灰度图像
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # 使用高斯滤波平滑图像
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        # 使用霍夫变换检测圆形
        circles = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, 1.2, 100)
        # 如果检测到圆形，绘制边界
        if circles is not None:
            # 将圆形坐标转换为整数
            circles = np.round(circles[0, :]).astype("int")
            # 定义一个阈值，表示两个圆心之间的最大距离，用于判断是否合并
            threshold = 10
            # 定义一个列表，用于存储合并后的圆形
            merged_circles = []
            # 遍历每个圆形
            for (x1, y1, r1) in circles:
                # 定义一个标志，表示当前的圆形是否已经被合并过
                merged = False
                # 遍历已经合并过的圆形列表
                for i in range(len(merged_circles)):
                    # 获取已经合并过的圆形的坐标和半径
                    (x2, y2, r2) = merged_circles[i]
                    # 计算两个圆心之间的距离
                    distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    # 如果距离小于阈值，说明两个圆形可以合并
                    if distance < threshold:
                        # 将当前的圆形和已经合并过的圆形进行平均，得到新的圆形
                        if r1 >= r2:
                            merged_circles[i] = (x1, y1, r1)
                        else:
                            merged_circles[i] = (x2, y2, r2)
                        # 设置标志为True，表示当前的圆形已经被合并过
                        merged = True
                        # 跳出循环，不再遍历其他已经合并过的圆形
                        break
                # 如果当前的圆形没有被合并过，就将它添加到已经合并过的圆形列表中
                if not merged:
                    merged_circles.append((x1, y1, r1))
            # 遍历已经合并过的圆形列表，找到最大的圆，作为检测结果
            biggest = None
            biggest_r = -1
            for (x, y, r) in merged_circles:
                if r > biggest_r:
                    biggest = (x, y, r)
                    biggest_r = r
            # 将圆转换成bbox
            if biggest is not None:
                self.biggest = biggest
                self.bbox = np.array([[biggest[0] - biggest[2], biggest[1] - biggest[2]],
                                      [biggest[0] + biggest[2], biggest[1] + biggest[2]]] + bias, np.int32)
                return self.__refine_bbox(self.bbox)
        return None

    def __refine_bbox(self, bbox):
        # refine bbox
        bbox[:, 0] = np.clip(bbox[:, 0], 0, self.image_width)
        bbox[:, 1] = np.clip(bbox[:, 1], 0, self.image_height)
        w, h = bbox[1] - bbox[0]
        if w <= 0 or h <= 0 or w * h <= self.min_area:
            return None
        else:
            return bbox
    
    def get_status(self, image):
        self.dashboard_status = None
        if self.bbox is not None:
            img_bgr = image
            
            img_rectangle = image[self.bbox[0][1]:self.bbox[1][1], self.bbox[0][0]:self.bbox[1][0], :]
            # 将图像转换为灰度图
            img_gray = cv2.cvtColor(img_rectangle, cv2.COLOR_BGR2GRAY)
            # 使用Canny边缘检测算法
            edges = cv2.Canny(img_gray, 50, 150)
            # 执行轮廓检测
            contours_rec, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 初始化最大面积和最大面积的矩形轮廓
            max_area = -1
            max_contour = None
            max_approx = None
            mid_x = None
            # 遍历所有轮廓
            for contour in contours_rec:
                # 计算轮廓的周长
                perimeter = cv2.arcLength(contour, True)
                # 近似轮廓，参数为轮廓周长的百分比（0.02表示轮廓周长的2%）
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                # 如果近似轮廓有4个顶点
                if len(approx) == 4:
                    # 计算轮廓的角度
                    angles = []
                    for i in range(4):
                        p1 = approx[i][0]
                        p2 = approx[(i + 1) % 4][0]
                        p3 = approx[(i + 2) % 4][0]
                    
                        v1 = p1 - p2
                        v2 = p3 - p2
                        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                        angles.append(np.degrees(angle))
                    # 判断轮廓的所有角是否接近于90度
                    if all(abs(angle - 90) < 10 for angle in angles):
                        # 计算矩形轮廓的面积
                        area = cv2.contourArea(contour)
                        
                        # 如果当前矩形轮廓的面积大于最大面积，则更新最大面积和最大面积的矩形轮廓
                        if area > max_area:
                            max_area = area
                            max_contour = contour
                            max_approx = approx
            
            # 如果找到了最大面积的矩形轮廓
            if max_area is not None:
                if (max_area / (self.biggest[2] ** 2)) < 0.04:
                    max_contour = None
            #print(max_area / (self.biggest[2] ** 2))
            if max_contour is not None:
                # 绘制最大面积的矩形轮廓
                point_1 = max_approx[0][0] + self.bbox[0]
                point_2 = max_approx[1][0] + self.bbox[0]
                point_3 = max_approx[2][0] + self.bbox[0]
                if np.linalg.norm(point_2 - point_1) > np.linalg.norm(point_2 - point_3):
                    mid_x = int((point_1[0] + point_2[0]) / 2)
                    mid_y = int((point_1[1] + point_2[1]) / 2)
                    
                    vector_ref_x = mid_x - self.biggest[0] 
                    vector_ref_y = mid_y - self.biggest[1]
                    cv2.line(image,(self.biggest[0],self.biggest[1]),(mid_x,mid_y),(255,0,0),1)
                else:
                    mid_x = int((point_1[0] + point_2[0]) / 2)
                    mid_y = int((point_1[1] + point_2[1]) / 2)
                    vector_ref_x = mid_x - self.biggest[0] 
                    vector_ref_y = mid_y - self.biggest[1]
                    cv2.line(image,(self.biggest[0],self.biggest[1]),(mid_x,mid_y),(255,0,0),1)
                #print(max_area / (self.biggest[2] ** 2))
                x, y, w, h = cv2.boundingRect(max_contour)
                x = x + self.bbox[0][0]
                y = y + self.bbox[0][1]
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.drawContours(img_rectangle, [max_approx], 0, (0, 0, 255), 2)
            
            # 取完整圆的60%部分
            center = np.mean(self.bbox, axis=0)
            #print(center)
            scale_bbox = center + (self.bbox - center) * 0.5
            scale_bbox = scale_bbox.astype(np.int32)
            image = image[scale_bbox[0][1]:scale_bbox[1][1], scale_bbox[0][0]:scale_bbox[1][0], :]
            
            
            
            
            
            
            
            #print(image.shape)
            # 图片转换为灰度图
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            # 二值化，将黑色区域设为255，其他区域设为0
            _, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)
            # 寻找轮廓
            contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            # 找到面积最大的轮廓
            max_area = 0
            max_contour = None
            for c in contours:
                area = cv.contourArea(c)
                if area > max_area:
                    max_area = area
                    max_contour = c


            # 如果找到了最大轮廓，绘制它并计算它的角度
            if max_contour is not None:
                
                
                rect = cv.minAreaRect(max_contour)
                # 得到指针轮廓的最佳拟合直线，并获取其斜率
                vx, vy, x0, y0 = cv.fitLine(max_contour, cv.DIST_L2, 0, 0.01, 0.01)
                # 调用函数找到直线上距离中心点最远的点
                center_x = image.shape[0] // 2
                center_y = image.shape[1] // 2
                farthest_point = find_points_on_line(max_contour, vx, vy, x0, y0, center_x, center_y)
                if farthest_point:
                    farthest_point = farthest_point + scale_bbox[0]
                    cv.line(img_bgr, (self.biggest[0],self.biggest[1]),farthest_point,(0,0,255),1)
                theta = None
                # 若识别到参照物
                if mid_x:
                    # 计算两个向量的角度
                    theta1 = np.arctan2(farthest_point[1] - self.biggest[1], farthest_point[0] - self.biggest[0])
                    theta2 = np.arctan2(mid_y - self.biggest[1], mid_x - self.biggest[0])
                    # 角度相减
                    theta = theta1 - theta2
                    theta = np.degrees(theta)
                    if theta < 0:
                        theta = theta + 360
                # 判断状况
                if theta is not None:
                    if 30 <= theta < 115:
                        self.dashboard_status = "low"
                    elif 115 <= theta <= 235:
                        self.dashboard_status = "mid"
                    elif 235 < theta <= 330:
                        self.dashboard_status = "high"

                    else:
                        self.dashboard_status = None
                    
                
                
                
        else:
            self.dashboard_status = None
        return self.dashboard_status

    def visualize(self, image, status, thickness=1):
        if self.bbox is not None:
            # Draw bounding box on original image (in color)
            cv.rectangle(image, self.bbox[0], self.bbox[1], (0, 255, 0), 2)
            if status is not None:
                cv.putText(image, status, (self.bbox[0][0], self.bbox[0][1] + 22 * thickness),
                           cv.FONT_HERSHEY_SIMPLEX, thickness, (0, 0, 255))
        return image
