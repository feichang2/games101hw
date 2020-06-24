# 项目说明
学习[games101](https://sites.cs.ucsb.edu/~lingqi/teaching/games101.html)图形学课程中,老师讲的很好,[作业](http://games-cn.org/forums/topic/allhw/)也很有意思
不过作业没注释,环境也较复杂(cmake,eigen库,opencv库,虽然在wsl里都可以`apt-get install`,不过加上依赖似乎有500MB了),编译起来四个核也慢

用python重写了一下作业代码:
- 尽量使用较少的库
- 添加上注释
- 减少数据结构的封装,暴露存储方式和运算,看起来逻辑更清楚


因为老师说不建议公布作业的代码,去掉了原来作业的代码

# 项目依赖
opencv
numpy
```bash
pip install opencv-python numpy
```
