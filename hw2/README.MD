# 说明

实验的代码在lab2.py里面

直接运行就行了


处理了opencv的显示问题:
- 创建帧和深度数组的时候shape为(height,width,3)而不是之前的(width,height,3)
- 绘制像素的时候会进行(x,y)到像素坐标(height-y,x)的转换
- opencv显示颜色使用的是BGR,所以将RGB颜色进行了一次转换`frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)`

