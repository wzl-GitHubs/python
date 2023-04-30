<style>Matplotlib模块
</style>

# 1 Matplotlib模块

## 1-1、Matplotlib的安装

使用pip安装方式：进入命令行cmd（win+r），在cmd窗口输入代码如下：

```
python -m pip install Matplotlib -i 镜像源（如下所示）
```

比如：清华大学镜像源：https://pypi.tuna.tsinghua.edu.cn/simple

## 1-2、Matplotlib库简单使用

##### A.      绘制包括5个元素列表的图形

```python
import matplotlib.pyplot as plt  
plt.plot([3,1,4,5,2])  
plt.ylabel(**'Grade'**) #增加y轴标签  
plt.show()
```

结论：plt.plot()只有一个输入列表或数组时，参数被当作Y轴， X轴以索引自动生成

##### B.       将绘制的图形保存

```python
import matplotlib.pyplot as plt  
plt.plot([3,1,4,5,2])  
plt.ylabel(**'Grade'**)  
plt.savefig(**'test'**,dpi=600) *#**默认输出为**PNG**文件*plt.show()
```

结论：plt.savefig()将输出图形存储为文件，默认PNG格式，可以通过dpi修改输出质量，其中dpi是指在每一英寸空间中包含的像素点的数量

##### C.       控制x轴和y轴

```python
import matplotlib.pyplot as plt  
plt.plot([0,2,4,6,8],[3,1,4,5,2])  
plt.ylabel(**'Grade'**)  
plt.axis([-1,10,0,6]) #控制横纵坐标尺寸  
plt.show()
```

结论：plt.plot(x,y)当有两个以上参数时，按照X轴和Y轴顺序绘制数据点

## 1-3、pyplot的绘图区域

pyplot可以在一个区域绘制两个及两个以上的图形，即分割绘图区域。分割绘图区域的办法主要是使用subplot，具体方法是：在全局绘图区域中创建一个分区体系，并定位到一个子绘图区域

subplot(nrows,ncols,plot_number)

plt.subplot(3,2,4)或者plt.subplot(324)

```python
import matplotlib.pyplot as plt  
import numpy as np  
def f(t):    return np.exp(-t)*np.cos(2*np.pi*t)  
a=np.arange(0.0,5.0,0.02)  
plt.subplot(211)  
plt.plot(a,f(a))  
plt.subplot(212)  
plt.plot(a,np.cos(2*np.pi*a),**'r--'**)  
plt.show()
```

# 2、pyplot的plot（）函数

plt.plot(x,y,format_string,**kwargs)

| 参数名称          | 参数说明                      |
| ------------- | ------------------------- |
| x             | x轴数据，列表或数据，可选             |
| y             | y轴数据，列表或数据                |
| format_string | 控制曲线的格式字符串，可选             |
| **kwargs      | 第二组或更多（x,y,format_string） |

注意：当绘制多条曲线时，各条曲线的x不能省略

## 2-1、绘制多条曲线

```python
import matplotlib.pyplot as plt  
import numpy as np  
a=np.arange(10)  
plt.plot(a,a*1.5,a,a*2.5,a,a*3.5,a,a*4.5)  
plt.show()
```

## 2-2、format_string：控制曲线的格式字符，由颜色字符、风格字符和标记字符组成。

```python
import matplotlib.pyplot as plt  
import numpy as np  
a=np.arange(10)  
plt.plot(a,a*1.5,**'go-'**,a,a*2.5,**'rx'**,a,a*3.5,**'*'**,a,a*4.5,**'b-.'**)  
plt.show()
```
