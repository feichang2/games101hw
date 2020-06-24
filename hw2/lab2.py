import cv2
import numpy as np
width, height = 700, 700
#创建一个width*height*3的数组,代表width*height的黑色(0,0,0)图片,也就是一帧
frame=np.zeros((height,width,3),dtype='uint8')
#depth数组放每个像素的深度值
depth=np.full((height,width),np.inf)
#创建一个窗口用来展示,可以适配图片大小,这是直接imshow做不到的
cv2.namedWindow("lab2")
#观测点
eye_pos=[0, 0, 5]
def get_model():
    #构建旋转,平移操作的矩阵
    #这里为单位矩阵
    return np.identity(4)
def get_view(eye_pos):
    #构建从世界坐标系到观测坐标系的矩阵
    return np.array([[1, 0, 0, -eye_pos[0]],[ 0, 1, 0, -eye_pos[1]], [0, 0, 1,
        -eye_pos[2]], [0, 0, 0, 1]],dtype=np.float)
def get_projection(fov,aspect,near,far):
    #构建进行透视投影的矩阵
    t2a=np.tan(fov/2.0)
    return np.array([[1./(aspect*t2a),0.,0.,0.],[0,1./t2a,0.,0.],[0.,0.,(near+far)/(near-far),2*near*far/(near-far)],[0.,0.,-1.,0.]])
def computeBarycentric(x,y,v):
    c1 = (x*(v[1][1] - v[2][1]) + (v[2][0] - v[1][0])*y + v[1][0]*v[2][1] - v[2][0]*v[1][1]) / (v[0][0]*(v[1][1] - v[2][1]) + (v[2][0] - v[1][0])*v[0][1] + v[1][0]*v[2][1] - v[2][0]*v[1][1])
    c2 = (x*(v[2][1] - v[0][1]) + (v[0][0] - v[2][0])*y + v[2][0]*v[0][1] - v[0][0]*v[2][1]) / (v[1][0]*(v[2][1] - v[0][1]) + (v[0][0] - v[2][0])*v[1][1] + v[2][0]*v[0][1] - v[0][0]*v[2][1])
    c3 = (x*(v[0][1] - v[1][1]) + (v[1][0] - v[0][0])*y + v[0][0]*v[1][1] - v[1][0]*v[0][1]) / (v[2][0]*(v[0][1] - v[1][1]) + (v[1][0] - v[0][0])*v[2][1] + v[0][0]*v[1][1] - v[1][0]*v[0][1])
    return (c1,c2,c3)
def inside_triangle(x,y,v):
    #判断点x,y是否在三角形v内
    #设三角形顶点为A,B,C,d里是向量BA,CB,AC
    d=[(v[1][0]-v[0][0],v[1][1]-v[0][1]),\
    (v[2][0]-v[1][0],v[2][1]-v[1][1]),\
    (v[0][0]-v[2][0],v[0][1]-v[2][1])]
    #s里放pA与BA,pB与CB,pC与AC叉乘的结果
    s=[]
    for i in range(3):
        s.append((x-v[i][0])*d[i][1]-(y-v[i][1])*d[i][0])
    #根据叉乘的结果符号是否一致判断是否在三角形内
    if(s[0]>0 and s[1]>0 and s[2]>0) or (s[0]<0 and s[1]<0 and s[2]<0):
        return True
    return False
def draw_triangle(frame,points,color,depth):
    #光栅化三角形
    #找到bounding_box的四个顶点值
    # print(points)
    width,height=frame.shape[:2]
    minx=min(points,key=lambda x:x[0])[0]
    maxx=max(points,key=lambda x:x[0])[0]
    miny=min(points,key=lambda x:x[1])[1]
    maxy=max(points,key=lambda x:x[1])[1]
    for i in range(minx,maxx+1):
        for j in range(miny,maxy+1):
            #遍历bounding_box的每个像素,计算是否在三角形中
            #采样点只选一个,为像素中心
            #计算像素对应的z深度
            alpha, beta, gamma = computeBarycentric(0.5+i, 0.5+j, points)
            w_reciprocal = 1.0/(alpha+ beta + gamma)
            z_interpolated = alpha * points[0][2]+ beta * points[1][2] + gamma * points[2][2]
            z_interpolated *= w_reciprocal
            if(z_interpolated < depth[height-j][i] and inside_triangle(0.5+i,0.5+j,points)):
                frame[height-j][i] = color
                depth[height-j][i] = z_interpolated

#两个三角形的顶点集合t
t=[[[2, 0, -2], [0, 2, -2], [-2, 0, -2]],[[3.5, -1, -5],[2.5, 1.5, -5],[-1, 0.5, -5]]]
#两个三角形的颜色集合
c=[(217.0, 238.0, 185.0),(185.0, 217.0, 238.0)]
#用来记录键盘输入的键值
key=0
f1 = (50 - 0.1) / 2.0;
f2 = (50 + 0.1) / 2.0;
while key != 27:
    #刷新一下帧和深度缓冲区
    frame.fill(0.)
    depth.fill(np.inf)
    #mvp变换矩阵
    mvp=get_projection(45, 1, 0.1, 50).dot(get_view(eye_pos)).dot(get_model())
    points=[]
    for i in range(len(t)):
        color=c[i]
        points.clear()
        for v in t[i]:
            d=mvp.dot(np.array(v+[1]))
            d/=d[3]
            #从[-1,1]到[width,height]的映射,[-1,1]到[0,2],乘上0.5就是[0,1],再乘上数组的边界
            points.append((int((d[0]+1.)*width*0.5),int((d[1]+1.0)*height*0.5),d[2]*f1+f2))
        draw_triangle(frame,points,color,depth)
    #因为imshow展示的是BGR的颜色,所以要转一下
    frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    cv2.imshow('lab2',frame)
    #等待10ms
    key=cv2.waitKey(10)
