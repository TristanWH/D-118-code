import os 
import time
import virtkey
import serial
import os
from time import sleep


serial_port = serial.Serial(
    port="/dev/ttyTHS0",
    baudrate=115200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
)

def server0():
    print("hello0")
    os.system("python3 final_start.py &")


def server1():
    print("hello1")
    os.system("python3 final.py &")

def server2():
    print("hello2")
    os.system("python3 final0.py &")

def server3():
    print("hello3")
    os.system("python3 final90.py &")




if __name__ == '__main__':
    v = virtkey.virtkey()
    flag=1
    while True:
        if serial_port.inWaiting() > 0:
            data = serial_port.read()
            data=data.decode("ascii")
            print(data)
            #serial_port.write(data)
            if data == 'a':
                server0()
            if data == 'c':
                server1()
            if data == 'd':
                server2()
            if data == 'f':
                server3()
            if data =='e':
                print("send esc")
                v.press_keysym(65307)
                time.sleep(0.5)
                v.release_keysym(65307)
            




    #wait里也可以设置按键，说明当按到该键时结
