import  cv2

import  mediapipe as mp
import time
#视频操作函数
#也可以导入视频
cap = cv2.VideoCapture(1)
#手部跟踪  处理的事RGB格式 所以使用 hands.process()处理的图像必须是RGB格式
myHands= mp.solutions.hands
hands= myHands.Hands()
mpDraw = mp.solutions.drawing_utils
pTime = 0
cTime = 0


while 1:
    #读取摄像头每一帧并显示
    success,img= cap.read()
    cv2.imshow("image",img)
    #必须是RGB格式 而得到的图像默认是BGR格式所以要转
    img_R=  cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result = hands.process(img_R)
    #检测所有手的列表,对列表进行访问可以获得 手的位置信息
    if(result.multi_hand_landmarks):
        for handLms in  result.multi_hand_landmarks:
            #每一个标志位都有一个id 获取并将其显示
            for id,lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                #获取界面中的坐标 ,这里经过测试是获取的小数需要*界面获取真正坐标
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv2.putText(img, str(int(id)), (cx , cy ), cv2.FONT_HERSHEY_PLAIN,
                            1, (0, 0, 255), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

