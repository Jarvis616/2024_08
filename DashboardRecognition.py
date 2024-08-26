import cv2 as cv
import numpy as np
import pytesseract
import cv2

# �ҵ������ϵ�������
def find_points_on_line(contour, vx, vy, x0, y0, center_x, center_y):
    line_points = []  # �������ֱ���ϵ�������

    # ��ȡֱ�ߵĵ�λ����
    unit_vector = np.squeeze(np.array([vx, vy]),axis = 1)
    #print(unit_vector)
    max_distance = 0  # ������Զ����
    farthest_point = None  # ����������ĵ���Զ�ĵ�
    #print(contour)
    for point in contour:
        # ��������ת��Ϊ numpy ����
        #print(point)
        point = np.array(point[0])
        #print(point)
        

        # ����ֱ���ϵ㵽�ο��������
        ref_point = np.squeeze(np.array([x0, y0]),axis = 1)
        #print(ref_point)
        vec_to_point = point - ref_point
        #print(vec_to_point)

        # ����㵽ֱ�ߵľ��루�㵽ֱ�ߵ�ͶӰ���ȣ�
        distance_to_line = (np.cross(vec_to_point, unit_vector))

        # ����һ����ֵ����ʾ����ֱ���ϵ����̷�Χ
        threshold = 0.1

        # ����㵽ֱ�ߵľ�������ֵ��Χ�ڣ�����Ϊ����ֱ����
        if distance_to_line < threshold:
            line_points.append(point)

            # ����㵽���ĵ�ľ���
            distance_to_center = np.linalg.norm(point - np.array([center_x, center_y]))

            # ������Զ�������Զ��
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
        # ת��Ϊ�Ҷ�ͼ��
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # ʹ�ø�˹�˲�ƽ��ͼ��
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        # ʹ�û���任���Բ��
        circles = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, 1.2, 100)
        # �����⵽Բ�Σ����Ʊ߽�
        if circles is not None:
            # ��Բ������ת��Ϊ����
            circles = np.round(circles[0, :]).astype("int")
            # ����һ����ֵ����ʾ����Բ��֮��������룬�����ж��Ƿ�ϲ�
            threshold = 10
            # ����һ���б����ڴ洢�ϲ����Բ��
            merged_circles = []
            # ����ÿ��Բ��
            for (x1, y1, r1) in circles:
                # ����һ����־����ʾ��ǰ��Բ���Ƿ��Ѿ����ϲ���
                merged = False
                # �����Ѿ��ϲ�����Բ���б�
                for i in range(len(merged_circles)):
                    # ��ȡ�Ѿ��ϲ�����Բ�ε�����Ͱ뾶
                    (x2, y2, r2) = merged_circles[i]
                    # ��������Բ��֮��ľ���
                    distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    # �������С����ֵ��˵������Բ�ο��Ժϲ�
                    if distance < threshold:
                        # ����ǰ��Բ�κ��Ѿ��ϲ�����Բ�ν���ƽ�����õ��µ�Բ��
                        if r1 >= r2:
                            merged_circles[i] = (x1, y1, r1)
                        else:
                            merged_circles[i] = (x2, y2, r2)
                        # ���ñ�־ΪTrue����ʾ��ǰ��Բ���Ѿ����ϲ���
                        merged = True
                        # ����ѭ�������ٱ��������Ѿ��ϲ�����Բ��
                        break
                # �����ǰ��Բ��û�б��ϲ������ͽ�����ӵ��Ѿ��ϲ�����Բ���б���
                if not merged:
                    merged_circles.append((x1, y1, r1))
            # �����Ѿ��ϲ�����Բ���б��ҵ�����Բ����Ϊ�����
            biggest = None
            biggest_r = -1
            for (x, y, r) in merged_circles:
                if r > biggest_r:
                    biggest = (x, y, r)
                    biggest_r = r
            # ��Բת����bbox
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
            # ��ͼ��ת��Ϊ�Ҷ�ͼ
            img_gray = cv2.cvtColor(img_rectangle, cv2.COLOR_BGR2GRAY)
            # ʹ��Canny��Ե����㷨
            edges = cv2.Canny(img_gray, 50, 150)
            # ִ���������
            contours_rec, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # ��ʼ�����������������ľ�������
            max_area = -1
            max_contour = None
            max_approx = None
            mid_x = None
            # ������������
            for contour in contours_rec:
                # �����������ܳ�
                perimeter = cv2.arcLength(contour, True)
                # ��������������Ϊ�����ܳ��İٷֱȣ�0.02��ʾ�����ܳ���2%��
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                # �������������4������
                if len(approx) == 4:
                    # ���������ĽǶ�
                    angles = []
                    for i in range(4):
                        p1 = approx[i][0]
                        p2 = approx[(i + 1) % 4][0]
                        p3 = approx[(i + 2) % 4][0]
                    
                        v1 = p1 - p2
                        v2 = p3 - p2
                        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                        angles.append(np.degrees(angle))
                    # �ж����������н��Ƿ�ӽ���90��
                    if all(abs(angle - 90) < 10 for angle in angles):
                        # ����������������
                        area = cv2.contourArea(contour)
                        
                        # �����ǰ���������������������������������������������ľ�������
                        if area > max_area:
                            max_area = area
                            max_contour = contour
                            max_approx = approx
            
            # ����ҵ����������ľ�������
            if max_area is not None:
                if (max_area / (self.biggest[2] ** 2)) < 0.04:
                    max_contour = None
            #print(max_area / (self.biggest[2] ** 2))
            if max_contour is not None:
                # �����������ľ�������
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
            
            # ȡ����Բ��60%����
            center = np.mean(self.bbox, axis=0)
            #print(center)
            scale_bbox = center + (self.bbox - center) * 0.5
            scale_bbox = scale_bbox.astype(np.int32)
            image = image[scale_bbox[0][1]:scale_bbox[1][1], scale_bbox[0][0]:scale_bbox[1][0], :]
            
            
            
            
            
            
            
            #print(image.shape)
            # ͼƬת��Ϊ�Ҷ�ͼ
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            # ��ֵ��������ɫ������Ϊ255������������Ϊ0
            _, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)
            # Ѱ������
            contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            # �ҵ������������
            max_area = 0
            max_contour = None
            for c in contours:
                area = cv.contourArea(c)
                if area > max_area:
                    max_area = area
                    max_contour = c


            # ����ҵ���������������������������ĽǶ�
            if max_contour is not None:
                
                
                rect = cv.minAreaRect(max_contour)
                # �õ�ָ��������������ֱ�ߣ�����ȡ��б��
                vx, vy, x0, y0 = cv.fitLine(max_contour, cv.DIST_L2, 0, 0.01, 0.01)
                # ���ú����ҵ�ֱ���Ͼ������ĵ���Զ�ĵ�
                center_x = image.shape[0] // 2
                center_y = image.shape[1] // 2
                farthest_point = find_points_on_line(max_contour, vx, vy, x0, y0, center_x, center_y)
                if farthest_point:
                    farthest_point = farthest_point + scale_bbox[0]
                    cv.line(img_bgr, (self.biggest[0],self.biggest[1]),farthest_point,(0,0,255),1)
                theta = None
                # ��ʶ�𵽲�����
                if mid_x:
                    # �������������ĽǶ�
                    theta1 = np.arctan2(farthest_point[1] - self.biggest[1], farthest_point[0] - self.biggest[0])
                    theta2 = np.arctan2(mid_y - self.biggest[1], mid_x - self.biggest[0])
                    # �Ƕ����
                    theta = theta1 - theta2
                    theta = np.degrees(theta)
                    if theta < 0:
                        theta = theta + 360
                # �ж�״��
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
