# 2024年中国高校智能机器人创意大赛---四足机器人专项赛

### 手势识别

1. 手势识别方法1：运行handpose1.py
2. 手势识别方法2：运行handpose2.py
3. 手势识别方法3：运行run_map_twice.py，为正式赛时使用的手势识别方法

### 智能巡检与应急救援
1. 运行run_map_twice.py，运行后按下“s”键先进行手势识别，然后有12s时间可以将机器狗手动搬到出发区，依次完成智能巡检与应急救援任务，完成一遍后将机器狗搬到起点完成第二遍
2. 仪表识别方法1：运行dashboard1.py,DashboardRecognition.py中包含仪表识别方法1所需函数
3. 仪表识别方法2：运行run_map_twice.py，为正式赛使用的仪表识别方法
3. 数字识别、锥形桶与小球识别均在run_map_twice.py中包含

### 附加赛任务：物料分拣
1. 运行material_sorting.py，运行后按下“s”键机器狗依次完成红色圆柱体定位、抓取并放入空纸箱动作

### yolo_models介绍
1. best_ball.pt为小球识别模型
2. best_dashboard.pt为仪表识别模型
3. best_hand.pt为手势识别模型
4. best_number.pt为数字识别模型
5. best_yzt.pt为圆柱体识别模型
6. best_zxt.pt为锥形桶识别模型
### 其他文件介绍
1. pytorchtrain.py为训练MLP的程序
2. train.py为训练YOLO模型的程序
3. hand_gestures_dataset_01.csv为MLP的输入数据集
4. best_gesture_recognition_model_7_4_01.pth为MLP分类手势模型
5. lenet_digit_classifier.pth为LeNet-5分类数字模型
6. YZGGPB.ttf为中文输出字体
7. angle_cx.py中包含所有PID控制相关函数
8. move_lite3包含机器狗发送指令码相关函数
9. rostopy.py包含对机器狗发布ros指令写成的函数
