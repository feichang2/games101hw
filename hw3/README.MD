添加了`objloader`库用来载入OBJ模型文件
直接`pip install objloader`即可

运行方式:
python3 lab3.py <output_filename> n[ormal]/p[hong]/t[exture]/b[ump]/d[isplacement] [hmap.jpg]

PS:hmap.jpg只有bump和displacement需要使用,是用来更换贴图的



一些存在的问题:
- 因为运算量较大,时间优势没有了
- bump和displacement渲染的时候,会有蓝色减少,红色增多的现象,原因大概是python的计算误差

