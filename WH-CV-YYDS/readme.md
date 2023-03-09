本作品原理是通过软件将两个usb摄像头在功能上变为网络摄像头，在同一网络环境下，访问摄像头对应的网址即可直接看到界面。

参考https://zhuanlan.zhihu.com/p/148536098

使用前请对应更改一下各个节点的网址

1. 主机先启动motion服务：sudo service motion start

2. 主机再启动motion: sudo motion

3. 首先在两个摄像节点打开“windows102.py”

4. 而后在主机上打开“controller.py”
