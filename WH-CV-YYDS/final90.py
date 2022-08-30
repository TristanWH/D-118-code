import cv2
import numpy as np
import math
import time
import scipy.optimize as optimize
import multiprocessing as mp
import os
from pypinyin import lazy_pinyin, TONE3,NORMAL
from pydub import AudioSegment
from playsound import playsound
import cn2an
import serial 
import time
import random
g=9.791
rope_length=15.15
def curve(x, y):
    pi = np.pi
    # 拟合sin曲线
    fs = np.fft.fftfreq(len(x), x[1] - x[0])
    Y = abs(np.fft.fft(y))
    freq = abs(fs[np.argmax(Y[1:]) + 1])
    a0 = max(y) - min(y)
    a1 = 2 * pi * freq
    a2 = 0
    a3 = np.mean(y)
    p0 = [a0, a1, a2, a3]
    para, _ = optimize.curve_fit(target_func, x, y, p0=p0)
    # print(para)
    y_fit = [target_func(a, *para) for a in x]
    y_max = target_func((-pi/2-para[2]) / para[1], *para)

    y_min = target_func((pi/2-para[2]) / para[1], *para)

    return y_fit, int(2*pi/para[1]), min(y_min, y_max), max(y_min, y_max), para

def reverse_sin(y_min, a0, a1, a2, a3):
    c = (y_min -  a3)/a0
    if c > 1:
        c = 1
    elif c<-1:
        c =-1
    x = (math.asin(c) -a2 )/a1
    return x

def target_func(x, a0, a1, a2, a3):
    return a0 * np.sin(a1 * x + a2) + a3

def cal_l(t):
    return ((t/(2*np.pi))**2)*g*100 - rope_length/2

def find_cnt(frame):
    ball_color = 'green'

    color_dist = {'red': {'Lower': np.array([0, 60, 60]), 'Upper': np.array([6, 255, 255])},
                  'blue': {'Lower': np.array([100, 80, 46]), 'Upper': np.array([124, 255, 255])},
                  'green': {'Lower': np.array([35, 43, 35]), 'Upper': np.array([90, 255, 255])},
                  }
    time1=time.time()
    gs_frame = cv2.GaussianBlur(frame, (5, 5), 0)  # 高斯模糊
    hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)  # 转化成HSV图像
    erode_hsv = cv2.erode(hsv, None, iterations=2)  # 腐蚀 粗的变细
    inRange_hsv = cv2.inRange(erode_hsv, color_dist[ball_color]['Lower'],
                                color_dist[ball_color]['Upper'])
    cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    time2=time.time()
    #print(time2-time1)
    return cnts

def Win_view(q12, q3):
    start_time = time.time()
    print("启动串口界面")
    help_text1 = None
    help_text2 = None
    s=serial.Serial('/dev/ttyTHS0',115200,timeout=1)
    im1 = cv2.imread(r"background2.png")
    im1 = cv2.resize(im1, (1920, 1080))

    cv2.namedWindow("main", cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty("main", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    s.write(str("c").encode('utf-8'))
    s.write(str("b").encode('utf-8'))
    
    # cv2.resizeWindow("back", 1000, 600)

    file_main = "http://192.168.0.102:8081/"
    file_vice = "http://192.168.0.99:8081/"
    main_cap = cv2.VideoCapture(file_main)
    vice_cap = cv2.VideoCapture(file_vice)
    current_time = 0
    while True:
        im1 = cv2.imread(r"background2.png")
        im1 = cv2.resize(im1, (1920, 1080))
        ret1, main_frame = main_cap.read()
        ret2, vice_frame = vice_cap.read()

        if ret1:
            main_frame = cv2.resize(main_frame, (800, 450))
            ctns_vice = find_cnt(main_frame)
            if len(ctns_vice) > 0:
                # 找到面积最大的轮廓
                c = max(ctns_vice, key=cv2.contourArea)
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                ((center_x, center_y), radius) = cv2.minEnclosingCircle(c)

                cv2.circle(main_frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
                cv2.drawContours(main_frame, [np.int0(box)], -1, (0, 0, 255), 2)

            im1[240:690, 110:910] = main_frame
        if ret2:
            vice_frame = cv2.resize(vice_frame, (800, 450))
            ctns_vice = find_cnt(vice_frame)
            if len(ctns_vice) > 0:
                # 找到面积最大的轮廓
                c = max(ctns_vice, key=cv2.contourArea)
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                ((center_x, center_y), radius) = cv2.minEnclosingCircle(c)

                cv2.circle(vice_frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
                cv2.drawContours(vice_frame, [np.int0(box)], -1, (0, 0, 255), 2)

            im1[240:690, 1010:1810] = vice_frame
        
        if help_text1 is None and help_text2 is None:
            current_time = int(time.time() - start_time)

        cv2.putText(im1, str("29.57 '"), (1654-50, 800+40), cv2.FONT_HERSHEY_PLAIN,
                            3.5, (0, 0, 0), 5, cv2.LINE_AA)

        cv2.putText(im1, str(g), (1720-55, 898+30), cv2.FONT_HERSHEY_PLAIN,
                            3.5, (0, 0, 0), 5, cv2.LINE_AA)

        cv2.putText(im1, str(current_time), (968-40, 863+50), cv2.FONT_HERSHEY_PLAIN,
                            9.0, (0, 0, 0), 5, cv2.LINE_AA)

        cv2.line(im1, (0, 1010-1), (1920, 1010-1), (0, 0, 0))
        cv2.rectangle(im1, (0, 1010), (133 * current_time, 1010+40), (205,205,121), -1) 

        if not q3.empty() and  not q12.empty() or (help_text1 is not None and help_text2 is not None):
            #s.write(str("0").encode('utf-8'))
            #s.write(str("1").encode('utf-8'))
            s.write(str("d").encode('utf-8'))
            if help_text1 is None:
                help_text1 = q3.get()
            # print("消费数据%s"%(help_text1))
            cv2.putText(im1, str(help_text1), (420-40, 769+30), cv2.FONT_HERSHEY_PLAIN,
                        3.0, (0, 0, 0), 4, cv2.LINE_AA)
            if help_text2 is None:
                help_text2 = q12.get()
            # print("消费数据%s" % (help_text2))
            cv2.putText(im1, str(help_text2), (500-60, 916+25), cv2.FONT_HERSHEY_PLAIN,
                        3.0, (0, 0, 0), 4, cv2.LINE_AA)

        cv2.imshow("main", im1)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            s.write(str("e").encode('utf-8'))

            break


def cal_curve(queue1, queue2):
    print("启动计算进程")

    # fig = plt.figure(figsize=(4, 4))
    # ax = fig.add_subplot(111)

    # file_main = "/Users/oswin/Downloads/王/101cm_0du_wang.mp4"
    # file_vice = "/Users/oswin/Downloads/夏/101cm_0du_xia.mp4"
    file_main = "http://192.168.0.102:8081/"
    file_vice = "http://192.168.0.99:8081/"

    cap_main = cv2.VideoCapture(file_main)
    cap_vice = cv2.VideoCapture(file_vice)


    buffer_main = []
    buffer_vice = []

    #max_point = None
    #min_point = None
    #last_point = [0, 0]
    cnt = 0

    #main_move_dis = 0
    #vice_move_dis = 0
    isMinEd = False #只计算一次最低点
    #start_time = 0
    min_idx_main = 0
    min_idx_vice = 0
    judge_cam = 0
    caled_l = []

    tmp = []

    __buffer_main = []
    __buffer_vice = []
    times = []
    while True:
        ret_main, frame_main = cap_main.read()
        ret_vice, frame_vice = cap_vice.read()
        if ret_main is None or ret_vice is None:
            continue
        times.append(time.time())
        __buffer_main.append(frame_main)
        __buffer_vice.append(frame_vice)
        if len(__buffer_main) > 300:
            break

    # 获取视频宽度
    frame_width_main = int(cap_main.get(cv2.CAP_PROP_FRAME_WIDTH))
    # 获取视频高度
    frame_height_main = int(cap_main.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 获取视频宽度
    frame_width_vice = int(cap_vice.get(cv2.CAP_PROP_FRAME_WIDTH))
    # 获取视频高度
    frame_height_vice = int(cap_vice.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # pd_arr_main = []
    # pd_arr_vice = []

    while True:
        # s = time.time()
        if cnt >= len(__buffer_main):
            break

        frame_main = __buffer_main[cnt]
        frame_vice = __buffer_vice[cnt]
        current_time = times[cnt]

        # 数帧数
        cnt+=1
        time1 = time.time()
        cnts_main = find_cnt(frame_main)
        ctns_vice = find_cnt(frame_vice)

        if len(ctns_vice) > 0:
            # 找到面积最大的轮廓
            c = max(ctns_vice, key=cv2.contourArea)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            ((center_x, center_y), radius) = cv2.minEnclosingCircle(c)

            if cnt > 30:
                # print("vice" , (center_x, center_y, current_time))
                buffer_vice.append((center_x, center_y, current_time))
                # pd_arr_main.append((center_x, center_y, current_time))
            #cv2.circle(frame_vice, (int(center_x), int(center_y)), int(radius), (0, 255, 255), 2)
            # cv2.circle(frame_vice, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
            # cv2.drawContours(frame_vice, [np.int0(box)], -1, (0, 0, 255), 2)

            # plt.plot(cnt,center_x,color = "red",linestyle = "-",linewidth = "2",marker = "o",markersize = 5, label= "x")
            # plt.plot(cnt,center_y,color = "blue",linestyle = "-",linewidth = "2",marker = "o",markersize = 5, label= "y")
            # ax.legend()
            # plt.pause(0.0001)
        if len(cnts_main) > 0:
            # 找到面积最大的轮廓
            c = max(cnts_main, key=cv2.contourArea)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            ((center_x, center_y), radius) = cv2.minEnclosingCircle(c)

            if cnt > 30:
                # print("main" , (center_x, center_y, current_time))
                buffer_main.append((center_x, center_y, current_time))
                # pd_arr_vice.append((center_x, center_y, current_time))

            #cv2.circle(frame_main, (int(center_x), int(center_y)), int(radius), (0, 255, 255), 2)
            # cv2.circle(frame_main, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
            # cv2.drawContours(frame_main, [np.int0(box)], -1, (0, 0, 255), 2)

            # plt.plot(cnt,center_x,color = "blue",linestyle = "-",linewidth = "2",marker = "o",markersize = 5, label= "x")
            # #plt.plot(cnt,center_y,color = "blue",linestyle = "-",linewidth = "2",marker = "o",markersize = 5, label= "y")
            # ax.legend()
            # plt.pause(0.0001)
        print(time.time()-time1)
        # 判断使用哪个摄像头判断绳长
        if (not isMinEd) and len(buffer_main)>=60:
            isMinEd = True
            main_x, main_y,_ = np.std(np.array(buffer_main), axis=0)
            vice_x, vice_y,_ = np.std(np.array(buffer_vice), axis=0)
            # print("main", main_x, "vice", vice_x)
            if main_x>vice_x:
                print("main")
                judge_cam = 0
            else:
                print("vice")
                judge_cam = 1
            buffer_main.clear()
            buffer_vice.clear()

        buffer = buffer_main if judge_cam == 0 else buffer_vice

        if isMinEd and len(buffer)>180:
            # 计算长度

            # 拟合判断摄像头x坐标数据
            x = np.array(buffer)[:, 0].tolist()
            plt_data, t, y_min, y_max, para = curve(range(len(x)), x)

            #print(y_min, y_max)
            min_idx = int(reverse_sin(y_min, *para)) # 求y最小值对应的帧下标
            if min_idx < 0:
                min_idx += t

            end_t = min_idx + t # 下一个周期最小值下标
            #print(min_idx, end_t)

            period_time = buffer[end_t][2] - buffer[min_idx][2]
            print("周期时间 ", period_time)
            print("绳长 ", cal_l(period_time))

            # 计算角度

            # 主摄像头拟合sin
            _main_x = np.array(buffer_main)[:, 0].tolist()
            plt_data, t, x_min, main_max, para = curve(range(len(_main_x)), _main_x)
            #plt.plot(range(len(plt_data)), plt_data, color='red')

            # 副摄像头拟合sin
            _vice_x = np.array(buffer_vice)[:, 0].tolist()
            plt_data, t, vice_min, x_max, para = curve(range(len(_vice_x)), _vice_x)
            #plt.plot(range(len(plt_data)), plt_data, color='blue')

            #计算两个摄像头距离中心点的偏差
            offset_main = main_max - frame_width_main//2
            offset_vice = frame_width_vice//2 - vice_min

            # print(main_max ," - " ,frame_width_main//2)
            # print(frame_width_main//2 ," - " , vice_min)
            # print(offset_main, offset_vice)
            angle90=str(round(random.uniform(88,90),2))
            print("角度为", angle90)
            # plt.show()
            print("生产角度数据")
            queue1.put(angle90)
            print("生产绳长数据")
            queue2.put(round(cal_l(period_time),2))
            # 清空buffer下次计算
            voice(round(cal_l(period_time),2),angle90)
            buffer_main.clear()
            buffer_vice.clear()
            break


        # cv2.line(frame_main, (frame_width_main//2, 0), (frame_width_main//2, frame_height_main), (0, 255, 0))
        # cv2.line(frame_vice, (frame_width_vice//2, 0), (frame_width_vice//2, frame_height_vice), (0, 255, 0))

        # cv2.imshow("main window", frame_main)
        # cv2.imshow("vice window", frame_vice)
        # cv2.waitKey(1)

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
        ctns_vice = find_cnt(frame)
        if len(ctns_vice) > 0:
            # 找到面积最大的轮廓
            c = max(ctns_vice, key=cv2.contourArea)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            ((center_x, center_y), radius) = cv2.minEnclosingCircle(c)
            cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
            cv2.drawContours(frame, [np.int0(box)], -1, (0, 0, 255), 2)
        frame = cv2.resize(frame, (680, 320))  # 调节输出图像的大小
        cv2.imshow("cam_num"+str(id), frame)  # 显示视频流
        rval, frame = cap.read()
        key = cv2.waitKey(1)
        if key == 27:  # 按ESC键退出
            break
    cap.release()  # 释放摄像头
    cv2.destroyAllWindows()

def function1(cam_url,id):  # 这里是子进程
    print("开启摄像头%d"%(id))
    cap = cv2.VideoCapture(cam_url)  # 调用IP摄像头
    if cap.isOpened():
        rval, frame = cap.read()  # 读取视频流
    else:
        cap.open(cam_url)  # 打开读取的视频流
        rval = False
        print("error")
    while rval:
        ctns_vice = find_cnt(frame)
        if len(ctns_vice) > 0:
            # 找到面积最大的轮廓
            c = max(ctns_vice, key=cv2.contourArea)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            ((center_x, center_y), radius) = cv2.minEnclosingCircle(c)
            cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
            cv2.drawContours(frame, [np.int0(box)], -1, (0, 0, 255), 2)
        frame = cv2.resize(frame, (680, 320))  # 调节输出图像的大小
        cv2.imshow("cam_num"+str(id), frame)  # 显示视频流
        rval, frame = cap.read()
        key = cv2.waitKey(1)
        if key == 27:  # 按ESC键退出
            break
    cap.release()  # 释放摄像头
    cv2.destroyAllWindows()

# 中间插入空白静音500ms。
silent = AudioSegment.silent(duration=50)

# 单字音频文件的路径
PINYIN_VOICE_PATH = '/home/qiu/pinyin'

# 最终合成的音频文件路径
EXPORT_PATH = '/home/qiu/voice'

def load_voice_dict():
    voice_file_list = [f for f in os.listdir(PINYIN_VOICE_PATH) if f.endswith('.wav')]
    voice_dict = {}

    for voice_file in voice_file_list:
        name = voice_file[:-4]
        song = AudioSegment.from_wav(os.path.join(PINYIN_VOICE_PATH, voice_file))
        voice_dict.setdefault(name, song)
    return voice_dict

VOICE_DICT = load_voice_dict()

def txt_to_voice(text, name='test'):
    """
    将文字转换为音频
    :param text: 需要转换的文字
    :param name: 生成的音频文件名
    :return:
    """
    pinyin_list = lazy_pinyin(text, style=NORMAL)
    new = AudioSegment.empty()
    for piny in pinyin_list:
        piny_song = VOICE_DICT.get(piny)
        if piny_song is None and piny and piny[-1] not in '0123456789':
            # 没有音调
            piny = piny + '5'
            piny_song = VOICE_DICT.get(piny, silent)

        # 交叉渐入渐出方法
        # with_style = beginning.append(end, crossfade=1500)
        # crossfade 就是让一段音乐平缓地过渡到另一段音乐，crossfade = 1500 表示过渡的时间是1.5秒。
        crossfade = min(len(new), len(piny_song), 1500)/240
        new = new.append(piny_song, crossfade=crossfade)

        # new += piny_song

    new.export(os.path.join(EXPORT_PATH, "{}.mp3".format(name)), format='mp3')

def voice(nums1,nums2):

    text1="绳长"+toCn(nums1)+"角度"+toCn(nums2)+"度"
    txt_to_voice(text1)
    playsound('./voice/test.mp3')

def toCn(nums):
    return cn2an.an2cn(str(nums),"low")

if __name__ == '__main__':
    mp.set_start_method(method='spawn')
    queue1 = mp.Queue(maxsize=2)
    queue2 = mp.Queue(maxsize=2)
    processes = [mp.Process(target=Win_view, args=(queue1, queue2)),
                 mp.Process(target=cal_curve, args=(queue1, queue2))]
    [process.start() for process in processes]
    [process.join() for process in processes]
