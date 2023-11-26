import cv2
import mediapipe as mp


class handDetector():  # 经典OOP
    # 设置初始条件
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands  # 最多同时出现几只手
        self.detectionCon = detectionCon  # 检测可信度
        self.trackCon = trackCon  # 跟踪可信度

        self.mpHands = mp.solutions.hands  # 用mediapipe找手
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    # 在图片里里找到手并返回这一帧图片
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 数字化视频输入
        self.results = self.hands.process(imgRGB)  # 处理视频找手

        if self.results.multi_hand_landmarks:  # 如果找到了手上的标识点
            for handLms in self.results.multi_hand_landmarks:
                if draw:  # 在识别出的手上把标记点画出来
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    # 找到手之后把手的关节位置投射上去并且记录数据
    def findPosition(self, img, handNo=0, draw=True):
        lmList = []  # 记录手上标识点
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            # 遍历识别数据，处理后输出
            for idNum, lm in enumerate(myHand.landmark):
                h, w, c = img.shape  # 高度，深度，通道数
                cx, cy = int(lm.x * w), int(lm.y * h)  # 坐标位置
                lmList.append([idNum, cx, cy])  # 可以在这里print一下看看长什么样
                if draw:  # 在识别出的点位置画个蓝点
                    cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
        return lmList


def main():
    wCam, hCam = 640, 480  # 摄像头拍摄大小
    cap = cv2.VideoCapture(0)  # 创建类用来拍摄
    cap.set(3, wCam)  # 比例设置
    cap.set(4, hCam)
    detector = handDetector(detectionCon=0.8)  # 最低准确度

    tipIds = [4, 8, 12, 16, 20]  # 指头的序号

    while True:
        success, img = cap.read()  # 获取一帧
        img = detector.findHands(img)  # 找手并返回标记好的图片
        lmList = detector.findPosition(img, draw=False)  # 标点然后返回数据

        if len(lmList) != 0:  # 如果找到了手且上面有标记好的点
            fingers = []

            # 大拇指的弯曲角度
            # 如果大拇指的第4个标记点像素位置低于第3个标记点，那它就是弯的
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # 其它四个指头角度判定
            for idNum in range(1, 5):
                if lmList[tipIds[idNum]][2] < lmList[tipIds[idNum] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            print(fingers)

            # 简单的手指二进制，用五根指头代表五位长的二进制数
            output = 0
            if fingers[0] == 1:  # 拇指竖起来
                output += 1
            if fingers[1] == 1:  # 食指竖起来
                output += 2
            if fingers[2] == 1:  # 中指竖起来
                output += 4
            if fingers[3] == 1:  # 无名指竖起来
                output += 8
            if fingers[4] == 1:  # 小指竖起来
                output += 16
            # 处理视频，画个方框，上面写识别到的数字
            cv2.rectangle(img, (20, 225), (250, 425), (0, 0, 0), cv2.FILLED)
            cv2.putText(img, str(output), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                        10, (255, 255, 255), 25)

        cv2.imshow("Image", img)  # 显示处理好的一帧图片
        cv2.waitKey(1)  # 相当于帧数了，这个是1ms一帧，1s60帧


if __name__ == "__main__":  # 这样就不会导入这个文件时直接跑程序啦
    main()