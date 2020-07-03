import cv2
import numpy as np
#创建一个列表用来放控制点
control_points = []

def mouse_handler(event,x,y,flags,param):
    #鼠标事件的绑定
    if(event==cv2.EVENT_LBUTTONDOWN and len(control_points)<4):
        #鼠标右键按下事件且控制点个数不满4时,添加控制点
        control_points.append(np.array([x,y]))

def naive_bezier(control_points,frame):
    #代数方法求贝塞尔曲线对应点
    p0,p1,p2,p3=control_points
    t=0.0
    while(t<=1.0):
        #各控制点乘上系数就是贝塞尔曲线上的点了
        p=(1-t)**3*p0+3*t*(1-t)**2*p1+3*t**2*(1-t)*p2+t**3*p3
        #BGR图像的序号2为红色对应位
        frame[int(p[1])][int(p[0])][2]=255
        t+=0.001
def recursive_bezier(control_points,t):
    #递归计算对应t的贝塞尔曲线点
    l=len(control_points)
    if(l==1):
        #递归结束条件,算到只有一个点了
        return control_points[0]
    #新的控制点数组
    new_points=[]
    for i in range(0,l-1):
        #计算新的控制点数组
        new_points.append(t*control_points[i+1]+(1-t)*control_points[i])
    #进行递归
    return recursive_bezier(new_points,t)
def bezier(control_points,frame):
    t=0.0
    while(t<=1.0):
        #几何计算对应t的贝塞尔曲线点
        p=recursive_bezier(control_points,t)
        #BGR图像的序号1为绿色对应位
        frame[int(p[1])][int(p[0])][1]=255
        t+=0.001

width, height = 700, 700
#创建一个width*height*3的数组,代表width*height的黑色(0,0,0)图片,也就是一帧
frame=np.zeros((width,height,3),dtype='uint8')
#创建一个窗口用来展示,可以适配图片大小,这是直接imshow做不到的
cv2.namedWindow("lab4")
#绑定方法到鼠标点击事件,用来输入控制点
cv2.setMouseCallback("lab4",mouse_handler)
#key用来记录键盘输入的键值
key=0
while key != 27:
    for point in control_points:
        #在每个控制点的位置画环
        cv2.circle(frame,tuple(point),3,(255,255,255),3)
    if(len(control_points)==4):
        #如果选择完了四个控制点
        #先使用naive_bezier画红色的贝塞尔曲线(这是写好了的,简单的代数运算)
        naive_bezier(control_points, frame)
        #再使用bezier画绿色的贝塞尔曲线(这是要自己写的,模拟几何运算)
        bezier(control_points, frame)
        #屏幕上展示一下
        cv2.imshow("lab4", frame)
        #输出为图片
        cv2.imwrite("output.png", frame)
        #等待关闭窗口
        key = cv2.waitKey(0)
        #直接退出
        break
    #屏幕上显示选点的结果
    cv2.imshow('lab4',frame)
    #等待10ms
    key=cv2.waitKey(10)




