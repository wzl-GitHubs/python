<style>numpy模块
</style>

## 一、numpy模块

### 1、numpy的安装

使用pip安装方式：进入命令行cmd（win+r），在cmd窗口输入代码如下：

python -m pip install numpy -i 镜像源（如下所示）

国内常用镜像源如下：

清华大学：https://pypi.tuna.tsinghua.edu.cn/simple

阿里云：http://mirrors.aliyun.com/pypi/simple/

中国科技大学 https://pypi.mirrors.ustc.edu.cn/simple/

华中理工大学：http://pypi.hustunique.com/

山东理工大学：http://pypi.sdutlinux.org/

豆瓣：http://pypi.douban.com/simple/

网易：http://mirrors.163.com

### 2、numpy的数据类型

布尔型、整数、无符号整数、浮点数和复数

数据类型转换：

```python
import numpy as np
print(np.float64(42))
print(np.int8(42.0))
```

### 3、数组对象ndarray

1）创建ndarray：调用numpy的array()函数，语法格式为

numpy.array(object,dtype=None,copy=None,order=None,subok=False,ndmin=0)，其中，数组的数据类型默认是浮点型

例：使用array函数创建一个ndarry

注意：使用array()函数创建一个ndarry时，需要将python列表作为参数，而列表中的元素就是ndarray的元素。

```python
import numpy as np  # 导入 numpy 库  
a=np.array([1,2,3,4,5])                 # 创建一维数组
b=np.array([[1,2,3,4],[4,5,6,7],[7,8,9,10]]) # 创建二维数组
输出数组：print(a)
         print(b)

# 查看数组维度：print(a.ndim)
print(b.ndim)

# 查看数组长度：print(a.size)
print(b.size)

# 查看数组类型：print(a.dtype)
print(b.dtype)

# 查看数组形状：print(a.shape)
print(b.shape)

# 修改数组的shape属性，在保持数组元素个数不变的情况下改变每个轴的长度。例如：

b.shape=4,3

print(b)
```

### 4、内置的数组创建方法

##### 1）arange函数

通过起始值、终止值和步长来创建一个一维数组，在创建的数组中并不包括终止值。语法格式如下：

numpy.arange([start,]stop,[step,]dype=None)

| 参数名称  | 说明                             |
| ----- | ------------------------------ |
| start | 数组的开始值，生成的数组包括该值，默认为0，接收int或实数 |
| stop  | 数组的终值，生成的数组不包括该值，接收int或实数      |
| step  | 在数组中，值之间的步长，默认为1，接收int或实数      |
| dtype | 数组的类型                          |

例如：

```python
import numpy as np  
a=np.arange(0,10,1,dtype=float)
print(a)
```

##### 2）zeros函数

用于创建元素全部为0的数组

例如：创建一个元素全为0的2行3列数组，要求数据类型为整型

```python
import numpy as np  
a=np.zeros((2,3),dtype=int)
print(a)
```

##### 3）linspace函数

通过开始值、终值和元素个数来创建一个一维数组，默认包括终值。语法格式如下：

numpy.linspace(start,stop,num=50,dtype=None)

| 参数名称  | 说明           |
| ----- | ------------ |
| start | 开始值          |
| stop  | 终值           |
| num   | 生成的样本数，接收int |
| dtype | 数组的类型        |

例如：创建一个开始值为0，终值为1的，包括12个元素的数组

```python
import numpy as np  
a=np.linspace(0,1,12)
print(a)
```

##### 4）eye函数

用于生成对角线元素为1，其他元素为0的数组，类似于对角矩阵

例如：

```python
import numpy as np  
a=np.eye(3)
print(a)
```

##### 5）logspace函数

与linspace函数类似，但它创建是等比数列，语法格式如下：

numpy.logspace(start,stop,num=50,base=10.0,dtype=None)

其中，开始值和终值默认都是10的幂，base可以设置日志空间的底数，在不设置的情况下，默认为10

例如：

```python
import numpy as np  
a=np.logspace(0,2,20)
print(a)
```

或者：

```python
import numpy as np  
a=np.logspace(0,2,20,base=2)
print(a)
```

##### 6）diag函数

用于创建除对角线上的元素以外的其他元素都为0的数组

例如：

```python
import numpy as np  
a=np.diag([1,2,3,4])
print(a)
```

### 5、生成随机数：random子模块

##### 1）rand()函数

用于生成一个任意维数的数组，数组的元素取自0-1上平均分布，语法格式如下：

numpy.random.rand(d0,d1,d2,...,dn)

其中，d0,d1,d2,...,dn表示数组的维度，必须是非负数。

例如：

```python
import numpy as np  
a=np.random.rand(5,10)
print(a)
```

##### 2）randint()函数

用于生成指定范围的随机数，语法格式如下：

numpy.random.randint(low,high=None,size=None,dtype=int)

| 参数名称  | 说明    |
| ----- | ----- |
| low   | 数组最小值 |
| high  | 数组最大值 |
| size  | 数组的形状 |
| dtype | 数组的类型 |

例如：

import numpy as np
a=np.random.randint(2,10,2)
print(a)

练习题：生成一个最小值不低于2，最大值不高于10的2行5列随机数组

import numpy as np
a=np.random.randint(2,10,size=(2,5))
print(a)

##### 3）random()函数

语法格式如下：
numpy.random.random(size=None)

其中，size表示返回的随机浮点数个数

例如：

```python
import numpy as np  
a=np.random.random(2)
print(a)
```

练习题：通过random函数生成一个3行4列的随机数组

```python
import numpy as np  
a=np.random.random((3,4))
print(a)
```

### 6、通过索引访问数组

##### 1）一维数组的索引

与list的索引方法一致

```python
import numpy as np  
a=np.arange(10)  
print(a[5]) *#**用整数作为索引获取数组中的某个元素*
print(a[3:5])  *#**用范围作为索引获取数组的一个切片*
print(a[:5])  *#**省略开始索引，表示从**a[0]**开始，不包括**a[5]*
print(a[-1])  *#**索引使用负数，**-1**表示从数组最后往前数的第一个元素*
a[2:4]=100,101  *#**修改元素的值*
print(a)
print(a[1:-1:2])  *#**带有步长的索引*
```

##### 2）多维数组的索引

多维数组的每一个轴都有一个索引，各个轴的索引之间用逗号隔开

例如：

```python
import numpy as np  
a=np.array([[1,2,3,4,5],[4,5,6,7,8],[7,8,9,10,11]])
print(a)
print(a[0,3:5]) *#**索引第**0**行第**3**、**4**列的元素*
print(a[1:,2:])  *#**索引第**1**、**2**行中第**2-4**列的元素*
print(a[:,2])  *#**索引第**2**列元素*
```

### 7、变换数组形状

##### 1）使用reshape函数改变数组的形状，同时，在改变数组的形状时，将改变数组的轴，语法格式如下：

numpy.reshape(a,newshape)

| 参数名称     | 说明        |
| -------- | --------- |
| a        | 需要变换形状的数组 |
| newshape | 变化后的形状    |

注意：reshape函数在改变原始数据的形状的同时不改变原始数据的值。如果指定的形状和数组的元素数目不匹配，那么函数将抛出异常

例如：

```python
import numpy as np  
a=np.arange(12) *#**创建一维数组*
print(a)
print(a.reshape(3,4)) *#**设置数组的形状*
```

##### 3）使用ravel（）函数完成数组展平

例如：

```python
import numpy as np  
a=np.arange(12).reshape(3,4)
print(a.ravel())
```

<style>
</style>

### 1、数组组合

##### 1）横向组合

hstack()函数用于实现水平（横向）组合（列方向）

例如：

```python
import numpy as np  
a=np.array([[1,2],[3,4]])  
b=np.array([[5,6],[7,8]])
print(a)
print(b)  
c=np.hstack((a,b))
print(c)
```

##### 2）纵向组合

vstack()函数用于实现垂直（纵向）组合（行方向）

例如：

```python
import numpy as np  
a=np.array([[1,2],[3,4]])  
b=np.array([[5,6],[7,8]])
print(a)
print(b)  
c=np.vstack((a,b))
print(c)
```

##### 3）concatenate()函数用于实现数组的横向组合和纵向组合，当参数axis=1时，横向组合，当参数axis=0时，纵向组合数组

例如：

```python
import numpy as np  
a=np.array([[1,2],[3,4]])  
b=np.array([[5,6],[7,8]]) 
print(a)
print(b)  
c=np.concatenate((a,b),axis=1)  *#**横向组合数组*
print(c)
```

### 2、数组分割

##### 1）hsplit()函数：对数组进行横向分割

例如：

```python
import numpy as np  
a=np.arange(16).reshape(4,4)
print(a)
print(np.hsplit(a,2))
```

##### 2）vsplit()函数：对数组进行纵向分割

例如：

```python
import numpy as np  
a=np.arange(16).reshape(4,4)
print(a)
print(np.vsplit(a,2))
```

##### 3）split()函数：实现数组的分割，当参数axis=1时，横向分割，当参数axis=0时，纵向分割

例如：

```python
import numpy as np  
a=np.arange(16).reshape(4,4)
print(a)
print(np.split(a,2,axis=1))  *#**横向分割数组*
```

### 3、创建numpy矩阵

##### 1）使用mat函数创建矩阵

```python
import numpy as np  
a=np.mat(**'1,2,3;4,5,6;7,8,9'**) *#**使用分号隔开数据*
print(a)
```

##### 2）使用matrix函数创建矩阵

```python
注意：调用mat函数和matrix函数等价
b=np.matrix([[1,2,3],[4,5,6],[7,8,9]])
print(b)
```

##### 3）bmat函数

可以将小矩阵组合成大矩阵

```python
a=np.eye(3)  
b=3*a
print(np.bmat(**'a b;a b'**))
```

##### 4）矩阵特有属性

```python
a=np.mat([[6,2,1],[1,5,2],[3,4,8]])
print(a.T)  *#**转置矩阵*
print(a.H)  *#**共轭转置矩阵*
print(a.I)  *#**逆矩阵*
```

##### 5）统计函数

###### A、amax函数获取最大值

```python
a=np.array([1,2,3,4,5,6,7,8,9])  
b=np.array([[1,3,5],[2,4,6],[8,10,12]])
print(np.amax(a))  *#**输出数组**a**中最大元素*
print(np.amax(b))  *#**输出数组**b**中最大元素*
print(np.amax(b,axis=0))  *#**纵向获取数组**b**中最大元素*
print(np.amax(b,axis=1))  *#**横向获取数组**b**中最大元素*
```

###### B、sum函数获取数组中所有元素的和

```python
import numpy as np  
a=np.array([1,2,3,4,5,6,7,8,9])  
b=np.array([[1,3,5],[2,4,6],[8,10,12]])
print(np.sum(a))  *#**输出数组**a**所有元素的和*
print(np.sum(b))  *#**输出数组**b**所有元素的和*
print(np.sum(b,axis=0))  *#**纵向获取数组**b**所有元素的和*
print(np.sum(b,axis=1))  *#**横向获取数组**b**所有元素的和*
```

###### C、ptp函数计算数组中最大值与最小值的差

```python
a=np.array([1,2,3,4,5,6,7,8,9])  
b=np.array([[1,3,5],[2,4,6],[8,10,12]])
print(np.ptp(a))  *#**输出数组**a**中元素差*
print(np.ptp(b))  *#**输出数组**b**中元素差*
print(np.ptp(b,axis=0))  *#**以纵向计算数组**b**中元素差*
print(np.ptp(b,axis=1))  *#**以横向计算数组**b**中元素差*
```

### 4、掌握ufunc函数

##### 1）常用的ufunc函数运算

###### A、四则运算

数组间的四则运算表示对数组中的每个元素分别进行四则运算

注意：进行四则运算的两个数组的形状必须相同

```python
import numpy as np  
x=np.array([1,2,3])  
y=np.array([4,5,6])  
print(x+y)  *#**数组相加*print(x-y)  *#**数组相减*print(x*y)  *#**数组相乘*print(x/y)  *#**数组相除*print(x**y) *#**数组幂运算*
```

###### B、比较运算

> 、< 、== 、>=、 <=、!=

比较运算返回的结果是一个布尔型数组，每个元素为数组对应元素的比较结果

```python
import numpy as np  
x=np.array([1,3,5])  
y=np.array([2,3,4])  
print(x>y)  
print(x<y)  
print(x==y)  
print(x!=y)
```

###### C、逻辑运算

numpy.all函数用于测试所有数组元素的计算结果是否为True

numpy.any函数用于测试任何数组元素的计算结果是否为True

```python
import numpy as np  
x=np.array([1,3,5])  
y=np.array([2,3,4])  
print(np.all(x==y))  
print(np.any(x==y))
```

##### 2）ufunc函数的广播机制

###### A、一维数组的广播

```python
import numpy as np  
arr1=np.array([[0,0,0],[1,1,1],[2,2,2],[3,3,3]])  
print(arr1)  
print(arr1.shape)  
arr2=np.array([1,2,3])  
print(arr2)  
print(arr2.shape)  
print(arr1+arr2)
```

| 0   | 0   | 0   |
| --- | --- | --- |
| 1   | 1   | 1   |
| 2   | 2   | 2   |
| 3   | 3   | 3   |

| 1   | 2   | 3   |
| --- | --- | --- |
| 1   | 2   | 3   |
| 1   | 2   | 3   |
| 1   | 2   | 3   |

| 1   | 2   | 3   |
| --- | --- | --- |
| 2   | 3   | 4   |
| 3   | 4   | 5   |
| 4   | 5   | 6   |

练习题：给出下列程序运行结果：

```python
import numpy as np  
arr1=np.array([[1,1,1],[2,2,2],[3,3,3]])
arr2=np.array([10,20,30])print(arr1*arr2)
```

###### B、二维数组的广播

| 1   | 1   | 1   |
| --- | --- | --- |
| 2   | 2   | 2   |
| 3   | 3   | 3   |
| 4   | 4   | 4   |

| 1   | 1   | 1   |
| --- | --- | --- |
| 3   | 3   | 3   |
| 5   | 5   | 5   |
| 7   | 7   | 7   |

```python
import numpy as np  
arr1=np.array([[0,0,0],[1,1,1],[2,2,2],[3,3,3]])  
print(arr1)  
print(arr1.shape)  
arr2=np.array([1,2,3,4]).reshape((4,1))  
print(arr2)  
print(arr2.shape)  
print(arr1+arr2)
```

| 0   | 0   | 0   |
| --- | --- | --- |
| 1   | 1   | 1   |
| 2   | 2   | 2   |
| 3   | 3   | 3   |

练习题：给出下列程序运行结果：

```python
import numpy as np  
arr1=np.arange(0,6).reshape(6,1)  
arr2=np.arange(0,5)  
print(arr1+arr2)
```

### 5、利用numpy进行统计分析

##### 1）读/写文件

###### A、save函数以二进制的格式保存数据，load函数从二进制文件中读取数据

save函数语法格式：

save(file,arr)

file:表示要保存的文件的名称，需要指定文件保存的路径

arr:表示需要保存的数组，即把arr保存到名称为file的文件中

```python
import numpy as np  
arr=np.arange(100).reshape(10,10)  
np.save('arrtest.npy',arr)  
print(arr)
```

###### B、如果要将多个数组保存到一个文件中，那么可以使用savez。

```python
import numpy as np  
arr1=np.array([[1,2,3],[4,5,6]])  
arr2=np.arange(0,1,0.1)  
np.savez('savez_arr.npz',arr1,arr2) *#**加载**npz**时会得到一个类似字典*
print(arr1,arr2,sep='\n')
```

###### C、当需要读取二进制文件时，可以使用load函数

```python
import numpy as nploaded_data=np.load('arrtest.npy')  
print(loaded_data)
```

注意：存储文件是可以省略扩展名，但读取时不能省略扩展名

D、读取含有多个数组的文件

```python
import numpy as np  
loaded_data1=np.load('savez_arr.npz')
print(loaded_data1['arr_0'],loaded_data1['arr_1'],sep='\n')
```

###### E、文本格式（txt/csv）文件的读写

可以使用savetxt函数、loadtxt函数和genfromtxt函数

A savetxt函数：可以将数组写到以某种分隔符隔开的文本文件中，语法格式如下：

numpy.savetxt(fname,X,delimiter=’’)

fname表示文件名，接收str

X表示数组数据

delimiter表示数据分隔符，接收str

A loadtxt函数：可以将文件中的数据加载到一个二维数组中，语法格式如下：

numpy.loadtxt(fname, delimiter=None)

fname表示需要读取的文件，接收str

delimiter表示分割数值的分割符，接收str

```python
import numpy as np  
arr=np.arange(0,12,0.5).reshape(4,6)  
print(arr)  
np.savetxt('arrtxt.txt',arr,fmt='%0.1f',delimiter=',')  
# fmt='%d'表示保存为整数*loaded_data=np.loadtxt(**'arrtxt.txt'**,delimiter=**','**)  
print(loaded_data)
```

A genfromtxt函数：面向的是结构化数组

通常使用的参数有3个：fname、delimiter和names

fname：表示用于存放数据文件的文件名

delimiter：表示用于分隔数据的字符

names：表示是否含有列标题的参数

```python
import numpy as np  
arr=np.arange(0,1.2,0.2).reshape(2,3)  
print(arr)  
np.savetxt('arrtxt.txt',arr,fmt='%0.1f',delimiter=',')  
loaded_data=np.genfromtxt('arrtxt.txt',delimiter=',')  
print(loaded_data)
```

注意：上述输出的结果是一组结构化的数据，names参数默认第一行为数据的列明，所以数据从第二行开始

##### 2）统计分析

###### A、排序

直接排序：对数值直接进行排序，使用sort函数

```python
import numpy as np  
np.random.seed(42) *#**设置随机种子*arr=np.random.randint(1,10,size=10)  
print(arr)  
arr.sort()  *#**直接排序*print(arr)  
np.random.seed(42)  
arr=np.random.randint(1,10,size=(3,3))  
print(arr)  
arr.sort(axis=1)  *#**横向排序*print(arr)  
arr.sort(axis=0)  *#**纵向排序*print(arr)
```

间接排序：根据一个或多个键对数据集进行排序，使用argsort函数和lexsort函数，使用这两个函数，可以得到一个由整数构成的索引数组，这些整数为排序后的元素的索引

argsort：

```python
import numpy as np  
arr=np.array([2,3,6,8,0,7])  
print(arr)  
print(arr.argsort())
```

| 0   | 1   | 2   | 3   | 4   | 5   |
| --- | --- | --- | --- | --- | --- |
| 2   | 3   | 6   | 8   | 0   | 7   |

| 0   | 2   | 3   | 6   | 7   | 8   |
| --- | --- | --- | --- | --- | --- |
| 4   | 0   | 1   | 2   | 5   | 3   |

lexsort：对多个数组进行排序时，按照最后一个数组计算

举个例子，对数字进行排序

```python
import numpy as np  
a=[0,4,0,3,2,3,3]  
b=[10,5,1,5,1,3,1]  
c=np.lexsort((a,b))  
print(c)
```

结果为：[2 0 4 6 5 3 1]

| 0   | 1   | 2   | 3   | 4   | 5   | 6   |
| --- | --- | --- | --- | --- | --- | --- |
| 0   | 4   | 0   | 3   | 2   | 3   | 3   |
| 10  | 5   | 1   | 5   | 1   | 3   | 1   |

| 0   | 0   | 2   | 3   | 3   | 3   | 4   |
| --- | --- | --- | --- | --- | --- | --- |
| 1   | 10  | 1   | 1   | 3   | 5   | 5   |
| 2   | 0   | 4   | 6   | 5   | 3   | 1   |

练习题：按照上述方法给出下列代码的运行结果：

```python
import numpy as np  
a=np.array([3,2,6,4,5])  
b=np.array([50,30,40,20,10])  
c=np.array([400,300,600,100,200])
d=np.lexsort((a,b,c))  
print(d)
```

###### B、去重和重复

unique函数：查找数组中的唯一值并返回已排序的结果

```python
import numpy as np  
names=np.array(['小明','小黄','小花','小明','小花','小兰','小白'])  
print(np.unique(names))
```

tile函数：可以将一个数据重复若干次，语法格式如下：

numpy.tile(A,reps)

A：表示输入的数组

reps：表示数组的重复次数

```python
import numpy as np  
arr=np.arange(5)  
print(np.tile(arr,3))
```

repeat函数：作用同tile函数，语法格式如下：

numpy.repeat(a,repeats,axis=None)

a:表示输入的数组

repeats：表示每个元素重复的次数

axis：表示沿着哪个轴进行重复

```python
import numpy as np  
np.random.seed(42)  
arr=np.random.randint(0,10,size=(3,3))  
print(arr)  
print(arr.repeat(2,axis=0))  
print(arr.repeat(2,axis=1))
```

思考：tile函数和repeat函数区别在哪？

注意：tile函数和repeat函数的主要区别在于：tile函数对数组进行重复操作，repeat函数对数组中的每个元素进行重复操作

## 课后练习题：

### 选择题：

##### (1)下列对Python中的NumPy描述不正确的是

A.NumPy是用于数据科学计算的基础模块

B.NumPy的数据容器能够保存任意类型的数据

C.NumPy提供了ndarray和array两种基本的对象

D.NumPy能够对多维数组进行数值运算

##### (2)下列选项中表示数组维度的是

A.ndim

B.shape

C.size

D.dtype

##### (3)代码" np . arange (0,1,0.2)"的运行结果为

A .[0.,0.2,0.4,0.6,0.8]

B .[0.,0.2,0.4,0.6,0.8,1.0]

C .[0.2,0.4,0.6,0.8]

D .[0.2,0.4,0.6,0.8,1.0]

##### (4)代码" np .linspace (0,10,5)"的运行结果为

A .[0,2.5,5,7.5]

B .[0,2.5,5,7.5,10]

C .[0.,2.5,5.,7.5.]

D .[0.,2.5.,5.,7.5.,10.]

##### (5)下列用于横向组合数组的函数是

A.hstack

B.hsplit

C.vstack

D.vsplit
