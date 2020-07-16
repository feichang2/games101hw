# 说明
这个光追实验和之前实验框架都不一样,将各个渲染对象作为了类进行实现,因为这样光追的主体程序逻辑更清晰
考虑自己实现向量的计算运算速度比不上numpy,以及直接二进制写ppm文件在windows上无法识别
还是使用了numpy进行向量的运算,以及使用opencv进行了图片的保存
保存语句在`lab5.py`的最后一行
`cv2.imwrite("./r.png",dst)`
修改为`cv2.imwrite("./binary.ppm",dst)`可以输出`binary.ppm`
# 依赖
numpy和opencv
# 运行
`python lab5.py`
等待约8分钟后,可以看到运行目录出现了一个`r.png`,里面就是渲染的结果
