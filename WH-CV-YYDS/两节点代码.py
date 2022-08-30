import cv2


def function2(cam_url,id):  # 这里是子进程
    print("开启摄像头%d"%(id))
    cap = cv2.VideoCapture(cam_url)  # 调用IP摄像头
    if cap.isOpened():
        rval, frame = cap.read()  # 读取视频流
    else:
        cap.open(cam_url)  # 打开读取的视频流
        rval = False
        print("error")
    while rval:
        # 获取视频宽度
        frame_width_main = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 获取视频高度
        frame_height_main = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cv2.line(frame, (frame_width_main // 2, 0), (frame_width_main // 2, frame_height_main), (0, 255, 0))
        out_win = "cam_num"+str(id)
        cv2.namedWindow(out_win, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(out_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(out_win, frame)
        rval, frame = cap.read()
        key = cv2.waitKey(1)
        if key == 27:  # 按ESC键退出
            break
    cap.release()  # 释放摄像头
    cv2.destroyAllWindows()

if __name__ =='__main__':
    function2("http://127.0.0.1:8081/",1)  # 正确做法：主线程只能写在 if内部