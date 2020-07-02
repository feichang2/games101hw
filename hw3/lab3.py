# coding=utf-8
import cv2
import numpy as np
import sys
from objloader import Obj
width, height = 700, 700
#创建一个width*height*3的数组,代表width*height的黑色(0,0,0)图片,也就是一帧
frame=np.zeros((height,width,3),dtype='float')
#depth数组放每个像素的深度值
depth=np.full((height,width),np.inf)
#观测点
eye_pos=[0, 0, 10]
def get_model(angle):
    #构建旋转,平移操作的矩阵
    s=np.sin(angle*np.pi/180.0)
    c=np.cos(angle*np.pi/180.0)
    #旋转矩阵
    rotation=np.array([[c,0.,s,0.],[0.,1.,0.,0.],[-s,0.,c,0.],[0.,0.,0.,1.]])
    #缩放矩阵
    scale=np.array([[2.5,0.,0.,0.],[0.,2.5,0.,0.],[0.,0.,2.5,0.],[0.,0.,0.,1.]])
    return rotation.dot(scale)
def get_view(eye_pos):
    #构建从世界坐标系到观测坐标系的矩阵
    return np.array([[1, 0, 0, -eye_pos[0]],[ 0, 1, 0, -eye_pos[1]], [0, 0, 1,
        -eye_pos[2]], [0, 0, 0, 1]],dtype=np.float)
def get_projection(fov,aspect,near,far):
    #构建进行透视投影的矩阵
    t2a=np.tan(fov/2.0)
    return np.array([[1./(aspect*t2a),0.,0.,0.],[0,1./t2a,0.,0.],[0.,0.,(near+far)/(near-far),2*near*far/(near-far)],[0.,0.,-1.,0.]])
def computeBarycentric(x,y,v):
    #中心坐标系的计算,第九次课才讲,这里不要求
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
def interpolate(alpha,beta,gamma,vert1,vert2,vert3):
    #插值函数
    if(len(vert1)==2):
        u = (alpha * vert1[0] + beta * vert2[0] + gamma * vert3[0])
        v = (alpha * vert1[1] + beta * vert2[1] + gamma * vert3[1])
        return np.array([u, v])
    return alpha*vert1+beta*vert2+gamma*vert3
def normalized(a):
    return a/np.linalg.norm(a)
def draw_triangle(frame,values,depth,fragment_shader):
    #光栅化三角形
    #找到bounding_box的四个顶点值
    points=[x[1] for x in values]
    width,height=frame.shape[:2]
    minx=max(0,int(min(points,key=lambda x:x[0])[0]))
    maxx=min(width-1,int(max(points,key=lambda x:x[0])[0])+1)
    miny=max(0,int(min(points,key=lambda x:x[1])[1]))
    maxy=min(height-1,int(max(points,key=lambda x:x[1])[1])+1)
    if(maxx<0 or minx>width-1 or maxy<0 or miny>height-1):
        #三角形的bounding_box不在观测范围内
        return
    for i in range(minx,maxx+1):
        for j in range(miny,maxy+1):
            #遍历bounding_box的每个像素,计算是否在三角形中
            #采样点只选一个,为像素中心
            #计算像素对应的z深度(根据中心坐标系对三个顶点的深度进行加权平均)
            alpha, beta, gamma = computeBarycentric(0.5+i, 0.5+j, points)
            w_reciprocal = 1.0/(alpha+ beta + gamma)
            z_interpolated = alpha * points[0][2]+ beta * points[1][2] + gamma * points[2][2]
            z_interpolated *= w_reciprocal
            if(z_interpolated < depth[height-j-1][i] and inside_triangle(0.5+i,0.5+j,points)):
                #更新深度
                depth[height-j-1][i] = z_interpolated
                #计算各个属性的插值
                interpolated_color = interpolate(alpha,beta,gamma,values[0][4],values[1][4],values[2][4])
                interpolated_normal = interpolate(alpha,beta,gamma,values[0][2],values[1][2],values[2][2])
                interpolated_uv = interpolate(alpha,beta,gamma,values[0][3],values[1][3],values[2][3])
                interpolated_xyz = interpolate(alpha,beta,gamma,values[0][0],values[1][0],values[2][0])
                #使用渲染函数计算对应像素的颜色
                frame[height-j-1][i] = fragment_shader.active_shader(interpolated_color,normalized(interpolated_normal),interpolated_uv,interpolated_xyz)
class Shader(object):
    #这是像素(片段)着色器
    #下面是一些默认的着色参数
    active_shader = None
    #记录各个光源的位置和光照强度
    lights = np.array([[[20, 20, 20], [500, 500, 500]],[[-20, 20, 0], [500, 500, 500]]])
    #环境光照的强度
    amb_light_intensity = np.array([10, 10, 10])
    #观测点坐标
    eye_pos = np.array([0,0,10])
    ka = np.array([0.005, 0.005, 0.005])
    ks = np.array([0.7937, 0.7937, 0.7937])
    p = 150
    texture = None
    def __init__(self,type='normal',texture_path=None):
        #初始化贴图
        if texture_path:
            self.texture = cv2.cvtColor(cv2.imread(texture_path),cv2.COLOR_BGR2RGB)
            self.height,self.width = self.texture.shape[:2]
        #根据命令行参数选择渲染函数
        if(type[0] == 'n'):
            self.active_shader = self.normal_fragment_shader
        elif(type[0] == 'p'):
            self.active_shader = self.phong_fragment_shader
        elif(type[0] == 't'):
            self.active_shader = self.texture_fragment_shader
        elif(type[0] == 'b'):
            self.active_shader = self.bump_fragment_shader
        elif(type[0] == 'd'):
            self.active_shader = self.displacement_fragment_shader
        else:
            pass
    def normal_fragment_shader(self,color,normal,uv,xyz):
        return (normal+np.array([1.,1.,1.]))/2.*255.
    def phong_fragment_shader(self,color,normal,uv,xyz):
        result_color = np.array([0.,0.,0.])
        for light in self.lights:
            L=light[0]-xyz
            r2=L.dot(L)
            l=normalized(L)
            v=normalized(self.eye_pos-xyz)
            h=normalized(l+v)
            ambient = self.ka * self.amb_light_intensity[0]
            diffuse = color * (light[1][0] / r2)*max(0.,normal.dot(l))
            specular = self.ks * (light[1][0] / r2)*pow(max(0.,normal.dot(h)),self.p)
            result_color += ambient+diffuse+specular
        return result_color*255.
    def texture_fragment_shader(self,color,normal,uv,xyz):
        kd=self.getColor(uv[0],uv[1])/255.
        result_color=np.array([0.,0.,0.])
        for light in self.lights:
            L=light[0]-xyz
            r2=L.dot(L)
            l=normalized(L)
            v=normalized(self.eye_pos-xyz)
            h=normalized(l+v)
            ambient = self.ka * self.amb_light_intensity[0]
            diffuse = kd * (light[1][0] / r2)*max(0.,normal.dot(l))
            specular = self.ks * (light[1][0] / r2)*pow(max(0.,normal.dot(h)),self.p)
            result_color += ambient+diffuse+specular
        return result_color*255.
    def bump_fragment_shader(self,color,normal,uv,xyz):
        #凹凸贴图的渲染
        kh = 0.2
        kn = 0.1
        x,y,z=normal
        t= np.array([x*y/np.sqrt(x*x+z*z),np.sqrt(x*x+z*z),z*y/np.sqrt(x*x+z*z)])
        b = np.cross(normal,t)
        tbn = np.array([t,b,normal]).T
        u,v=uv
        w = self.width
        h = self.height
        dU = kh * kn * (np.linalg.norm(self.getColor(u+1./w,v))-np.linalg.norm(self.getColor(u,v)))
        dV = kh * kn * (np.linalg.norm(self.getColor(u,v+1./h))-np.linalg.norm(self.getColor(u,v)))
        ln=np.array([-dU,-dV,1.])
        return normalized(tbn.dot(ln.T))*255.
    def displacement_fragment_shader(self,color,normal,uv,xyz):
        kd=color
        kh = 0.2
        kn = 0.1
        x,y,z=normal
        t= np.array([x*y/np.sqrt(x*x+z*z),np.sqrt(x*x+z*z),z*y/np.sqrt(x*x+z*z)])
        b = np.cross(normal,t)
        tbn = np.array([t,b,normal]).T
        u,v=uv
        w = self.width
        h = self.height
        dU = kh * kn * (np.linalg.norm(self.getColor(u+1./w,v))-np.linalg.norm(self.getColor(u,v)))
        dV = kh * kn * (np.linalg.norm(self.getColor(u,v+1./h))-np.linalg.norm(self.getColor(u,v)))
        ln=np.array([-dU,-dV,1.])
        xyz += kn * normal * np.linalg.norm(self.getColor(u,v))
        normal = normalized(tbn.dot(ln.T))
        result_color=np.array([0.,0.,0.])
        for light in self.lights:
            L=light[0]-xyz
            r2=L.dot(L)
            l=normalized(L)
            v=normalized(self.eye_pos-xyz)
            h=normalized(l+v)
            ambient = self.ka * self.amb_light_intensity[0]
            diffuse = kd * (light[1][0] / r2)*max(0.,normal.dot(l))
            specular = self.ks * (light[1][0] / r2)*pow(max(0.,normal.dot(h)),self.p)
            result_color += ambient+diffuse+specular
        return result_color*255.
    def getColor(self,u,v):
        #获取贴图对应坐标的颜色值
        u=np.fabs(u)
        v=np.fabs(v)
        if(u>1.):
            u=1.
        if(v>1.):
            v=1.
        u_img = u * self.width
        v_img = (1 - v) * self.height
        return self.texture[int(v_img)][int(u_img)]




#默认的输出图片文件名
if(len(sys.argv)>1):
    filename = sys.argv[1]
else:
    filename = "output.png"
#模型加载目录
obj_path = "./models/spot/"
#创建一个像素着色器,用来计算具体像素的值
if(len(sys.argv)>3):
    fragment_shader=Shader(sys.argv[2],obj_path+sys.argv[3])
else:
    fragment_shader=Shader(sys.argv[2],obj_path+"spot_texture.png")
#一个数组,记录了模型的各个顶点,每个顶点的信息(分先后)为顶点坐标,法向量坐标,纹理坐标(加了一个0.扩展到三维)
obj = Obj.open(obj_path+"spot_triangulated_good.obj").to_array()
#用来记录键盘输入的键值
key=0
#旋转变换的角度
angle=140.
f1 = (50 - 0.1) / 2.0;
f2 = (50 + 0.1) / 2.0;
#只经过了modle和view变换
mv=get_view(eye_pos).dot(get_model(angle))
#model和view变换的矩阵的逆矩阵的转置矩阵
inv_trans=np.linalg.inv(mv).T
#mvp变换矩阵
mvp=get_projection(45, 1, 0.1, 50).dot(mv)
points=[]
for i in range(0,len(obj),3):
    points.clear()
    for v in obj[i:i+3]:
        #每次取三个顶点数据作为三角形
        #临时存放数据的数组
        tmp=[]
        d=mv.dot(np.append(v[:3],[1]))
        #添加没有经过投影变换的顶点坐标
        tmp.append(d[:3])
        d=mvp.dot(np.append(v[:3],[1]))
        d/=d[3]
        #添加经过投影变换的顶点坐标数据
        tmp.append(np.array([(d[0]+1.)*width*0.5,(d[1]+1.0)*height*0.5,d[2]*f1+f2]))
        d=inv_trans.dot(np.append(v[3:6],[0.]))
        #添加经过变换后的顶点的法向量
        tmp.append(d[:3])
        #添加纹理坐标
        tmp.append(v[6:8])
        #添加颜色
        tmp.append(np.array([148,121.0,92.0])/255.)
        points.append(tmp)
    draw_triangle(frame,points,depth,fragment_shader)
dst=np.zeros((height,width,3),dtype=np.uint8)
#对负数进行归零,模拟convertTo函数
frame[frame<0.]=0.
frame=cv2.convertScaleAbs(frame,dst,1.0)
#因为imshow展示的是BGR的颜色,所以要转一下
dst=cv2.cvtColor(dst,cv2.COLOR_RGB2BGR)
cv2.imwrite(filename,dst)
