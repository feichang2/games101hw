添加了`objloader`库用来载入OBJ模型文件
直接`pip install objloader`即可

运行方式:
python3 lab3.py <output_filename> n[ormal]/p[hong]/t[exture]/b[ump]/d[isplacement] [hmap.jpg]

PS:hmap.jpg只有bump和displacement需要使用,是用来更换贴图的



一些存在的问题:
- 因为运算量较大,时间优势没有了
- bump和displacement渲染的时候,会有蓝色减少,红色增多的现象,~~原因大概是python的计算误差~~(误差来源于原实验使用的是convertTo进行数据格式转换,我这里使用的是convertScaleAbs进行的(因为python的opencv封装里没有这个前面那个),而前者将负数转成0,后者则是取绝对值,就是这里出现了问题,已经添加了`frame[frame<0.]=0.`进行转换前的预处理,该问题已解决)

