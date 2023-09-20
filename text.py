import sys
import cv2
import dlib
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import QTimer, QRect
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QPlainTextEdit, QListWidget
from PyQt5.QtGui import QPixmap, QImage
from face_recognition.predict import *
from face_detection.face_detect import my_face_detection as fd

class ImageClassifierWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.cap = cv2.VideoCapture(0)

    def initUI(self):
        # 设置窗口标题和大小
        self.setWindowTitle('寝室人员分类识别系统')
        self.setFixedSize(800, 500)

        # 初始化标签控件和按钮控件
        self.label_image = QLabel(self)
        self.label_image.move(20, 20)
        self.label_image.resize(350, 350)

        self.label_result = QLabel('结果:', self)
        self.label_result.move(400, 20)
        self.label_result.resize(100, 30)

        # self.list_widget = QListWidget(self)
        # self.list_widget.setGeometry(QtCore.QRect(400, 50, 350, 320))

        self.text_result = QPlainTextEdit(self)
        self.text_result.move(400, 50)
        self.text_result.resize(350, 320)

        self.button_load = QPushButton('使用摄像头', self)
        self.button_load.move(20, 400)
        self.button_load.resize(120, 30)

        self.button_predict = QPushButton('开始识别', self)
        self.button_predict.move(150, 400)
        self.button_predict.resize(120, 30)

        self.button_stop = QPushButton('停止', self)
        self.button_stop.move(280, 400)
        self.button_stop.resize(120, 30)
        # 连接信号和槽函数
        self.button_load.clicked.connect(self.loadImage)
        self.button_predict.clicked.connect(self.predictImage)
        self.button_stop.clicked.connect(self.stopCamera)

        # 数据预处理
        self.data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # 加载模型
        self.detector = dlib.get_frontal_face_detector()


        # 控件就会根据其内容自动进行缩放，以便更好地显示其内容
        self.label_image.setScaledContents(True)

        # 初始化人员名单
        self.attendance_list = ['li junqi', 'lu guoqi', 'mao wenchao', 'tian hanwen', 'xiao hongyin', 'zhou xingyu']
        self.info_dict = {'li junqi': {'姓名': '李俊奇', '学号': '190901109', '班级': '计算机科学与技术班'}, 'lu guoqi': {'姓名': '陆国奇', '学号': '190901115', '班级': '计算机科学与技术班'},
                     'mao wenchao':{'姓名': '毛文超', '学号': '190901118','班级': '计算机科学与技术班'},'tian hanwen':{'姓名': '田涵文', '学号': '190901122','班级': '计算机科学与技术班'},
                     'xiao hongyin': {'姓名': '肖鸿银', '学号': '190901128', '班级': '计算机科学与技术班'},'zhou xingyu':{'姓名': '周星宇', '学号': '190901137','班级': '计算机科学与技术班'}
                     }
    def loadImage(self):
        self.timer_camera = QTimer()
        self.timer_camera.timeout.connect(self.showImage)
        self.timer_camera.start(100)

    def showImage(self):
        # 获取摄像头每一帧
        _, frame = self.cap.read()
        # 将图像转换为Qt可读的格式
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 将图像显示到QLabel中
        self.label_image.setPixmap(QPixmap.fromImage(qt_image))

    def predictImage(self):
        # 初始化帧计数器
        frame_count = 0
        names = []  # 定义空列表来存储所有人的名字
        num_attendance = 0  # 记录出勤人数
        result_str = ''  # 记录识别到的同学所对应的信息
        result_name= ''  # 记录中文名字
        # 读取摄像头的一帧图像
        ret, frame = self.cap.read()

        # 把当前帧保存到指定的路径中
        cv2.imwrite('./JPEGImages/frame.jpg', frame)

        # 读取保存到指定路径的图像并识别人脸
        img_name = './JPEGImages/frame.jpg '


        # 检测人脸
        face_rects = fd(img_name)
        for dlib_rect in face_rects:
            # 裁剪人脸图像并识别
            (x, y, w, h) = (dlib_rect.left(), dlib_rect.top(), dlib_rect.right() - dlib_rect.left(), dlib_rect.bottom() - dlib_rect.top())
            face_img = cv2.imread(img_name)
            face_img = face_img[y:y+h, x:x+w]
            img = Image.fromarray(np.uint8(np.clip(face_img * 255, 0, 255)))
            img = self.data_transform(img)
            img = torch.unsqueeze(img, dim=0)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            json_path = './face_recognition/class_indices.json'
            assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

            with open(json_path, "r") as f:
                class_indict = json.load(f)
            # create model
            model = AlexNet(num_classes=6).to(device)
            # load model weights
            weights_path = "./face_recognition/AlexNet.pth"
            assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
            model.load_state_dict(torch.load(weights_path))
            model.eval()
            with torch.no_grad():
                # predict class
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()
                # 获取识别到的人的名字
                name = class_indict[str(predict_cla)]
                info = self.info_dict.get(name, {})  # 从字典中获取该学生的信息，如果没有则返回空字典
                result_name += '{},'.format(info.get('姓名', ''))
                result_str += '{}，学号：{}，班级：{}\n'.format(info.get('姓名', ''), info.get('学号', ''),
                                                            info.get('班级', ''))
                # 判断是否出勤
                if name in self.attendance_list:
                    num_attendance += 1
        # 获取所有人的中文姓名
        for student in self.info_dict.values():
            names.append(student['姓名'])

        # 计算未出席名单
        absence_names = [name for name in names if name not in result_name.split(',')]
        num_absence = len(absence_names)
        absence_str = ', '.join(absence_names)
        # self.label_result.setText(attendance_str)
        self.label_result.setText('结果')
        self.text_result.setPlainText(
            '寝室人员分类识别系统: {}/{}\n识别名单: {}\n未识别名单: {}\n未识别人数:{}\n已识别的宿舍人员信息: {}'.format(num_attendance, len(self.attendance_list),
                                                                            result_name, absence_str, num_absence, result_str))

        # 删除保存的图像
        os.remove(img_name)

    def stopCamera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if hasattr(self, 'timer_camera'):
            self.timer_camera.stop()

if __name__ == '__main__':
    # 创建应用程序和窗口
    app = QApplication(sys.argv)
    window = ImageClassifierWindow()
    window.show()

    # 运行窗口程序
    sys.exit(app.exec_())