import cv2
import numpy as np
width, height = 700, 700
#创建一个width*height*3的数组,代表width*height的黑色(0,0,0)图片,也就是一帧
frame=np.zeros((width,height,3),dtype='uint8')
#创建一个窗口用来展示,可以适配图片大小,这是直接imshow做不到的
cv2.namedWindow("lab1")
#模型旋转的角度
angle=0.0
#观测点
eye_pos=[0, 0, 5]
def get_model(angle):
    #构建旋转,平移操作的矩阵
    #这里只是绕z旋转angle
    s=np.sin(angle*np.pi/180.0)
    c=np.cos(angle*np.pi/180.0)
    return np.array([[c,-s,0.,0.],[s,c,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
def get_view(eye_pos):
    #构建从世界坐标系到观测坐标系的矩阵
    return np.array([[1, 0, 0, -eye_pos[0]],[ 0, 1, 0, -eye_pos[1]], [0, 0, 1,
        -eye_pos[2]], [0, 0, 0, 1]],dtype=np.float)
def get_projection(fov,aspect,near,far):
    #构建进行透视投影的矩阵
    t2a=np.tan(fov/2.0)
    return np.array([[1./(aspect*t2a),0.,0.,0.],[0,1./t2a,0.,0.],[0.,0.,(near+far)/(near-far),2*near*far/(near-far)],[0.,0.,-1.,0.]])
#三角形的顶点集合t
t=[[2, 0, -2], [0, 2, -2], [-2, 0, -2]]
#用来记录键盘输入的键值
key=0
while key != 27:
    #刷新一下帧
    frame.fill(0.)
    #mvp变换矩阵
    mvp=get_projection(45, 1, 0.1, 50).dot(get_view(eye_pos)).dot(get_model(angle))
    points=[]
    for v in t:
        d=mvp.dot(np.array(v+[1]))
        d/=d[3]
        #从[-1,1]到[width,height]的映射,[-1,1]到[0,2],乘上0.5就是[0,1],再乘上数组的边界
        points.append((int((d[0]+1.)*width*0.5),int((d[1]+1.0)*height*0.5)))
    #print(points)
    n=len(points)
    for i in range(n):
        for j in range(i+1,n):
            cv2.line(frame,points[i],points[j],(255,255,255),1)
    cv2.imshow('lab1',frame)
    #等待10ms
    key=cv2.waitKey(10)
    if key == ord('a'):
        angle +=10
    elif key == ord('d'):
        angle -=10
    else:
        pass


