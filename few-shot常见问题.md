# 代码调试常见问题

### **miniimagenet**数据集路径：

下列路径都是**miniimagenet数据集**在服务器上的**绝对路径**

```python
# csv
F:\datasets\mini-imagenet
    
# pkl
F:\datasets\mini-imagenet-pkl
    mini-imagenet-cache-train.pkl
    mini-imagenet-cache-val.pkl
    mini-imagenet-cache-test.pkl
    

```

![image-20210828155321099](https://gitee.com/lin2000h/images/raw/master/image-20210828155321099.png)

下面的为**miniimagenet**数据集的**train.csv**（里面存放的是**图片名称**以及对应的**标签**），这里注意下**第一行**，从第二行开始才是图片名称及其对应的标签。

![image-20210828155330091](https://gitee.com/lin2000h/images/raw/master/image-20210828155330091.png)



```python
# miniimagenet     pickle文件
F:\datasets\miniImageNet_load
```

![image-20210828155338092](https://gitee.com/lin2000h/images/raw/master/image-20210828155338092.png)

这里看了下 miniImageNet_category_split_train_phase_train.pickle 这个文件的结构；大致如下图:

![image-20210828155347764](https://gitee.com/lin2000h/images/raw/master/image-20210828155347764.png)

**catname2label：**类别数为64        **labels：**标签 也就是每个类600张图片 对应的有600个标签

```python
# 查看pickle文件的代码
with open('F:\datasets\miniImageNet_load\miniImageNet_category_split_train_phase_train.pickle','rb') as f:
    res = pickle.load(f, encoding='iso-8859-1')
print(res)
print(len(res['catname2label']))
print(len(res['labels']))
```



### gpu 



```python
# 在程序刚开始加这条语句可以提升一点训练速度，没什么额外开销
torch.backends.cudnn.benchmark = True 

#  按照PCI_BUS_ID顺序从0开始排列GPU设备 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

#设置当前使用的GPU设备仅为'args.gpu'(这里gpu=0)号设备  设备名称为'/gpu:0'
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
```

==这里设置workers = 0，也就是并行进程一般设置为0，这边服务器其实最大可以设置为8还是16，记不太清楚==



#### AttributeError: Can't pickle local object 'get_dataset.<locals>.<lambda>'

出现这个问题 则设置 **dataloader** 里面所有 **num_workers=0** 

### **contiguous()**

**个人总结：**

​			也就是内存存储多维数组的问题，python中**Tensor**实现的底层是**C**，所以多维数组是按照**行优先**存储的；当通过**transpose()、permute()**等方法修改数组维度以后，数组在内存中存储的顺序还是行优先，但是在语义上（逻辑上）维度发生改变，也就是相邻元素发生了改变。使用**contiguous()**方法后可以让修改后的数组在内存中**开辟一块新的内存空间**，保证修改后的数组的**语义相邻**，**内存**空间上**也相邻**。



**RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.**

```python
# 相应的地方添加 .contiguous()

# correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)

correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
```

**详细讲解**（https://zhuanlan.zhihu.com/p/64551412）



### **LongTensor**

Expected object of scalar type Long but got scalar type Int for argument #2 'target' in call to _thnn_nll_loss_forward

```python
# label = label.type(torch.cuda.LongTensor)

target = target.type(torch.cuda.LongTensor)
```



### cuda相关

**cuda第一次需要的激活问题**，每次其实都需要激活一下cuda。

![image-20210828155451230](https://gitee.com/lin2000h/images/raw/master/image-20210828155451230.png)

**解决：**

```python
# 输出cuda是否可用
print(torch.cuda.is_available())
```

![image-20210828155457894](https://gitee.com/lin2000h/images/raw/master/image-20210828155457894.png)

### **‘系统找不到指定的路径’**

**解释：**

原来有一个[`code-runner.respectShebang`](https://marketplace.visualstudio.com/items?itemName=formulahendry.code-runner)代码运行程序扩展名的设置，默认为`true`，但可以将其设置为false以允许您在脚本中保留shebang，但在通过代码运行程序运行代码时不使用它：

[]: https://www.pythonheidong.com/blog/article/389587/becd2b27d058514ec9af/

![image-20210828155505146](https://gitee.com/lin2000h/images/raw/master/image-20210828155505146.png)

```json
// settings.json中配置如下代码
"code-runner.respectShebang": false,
```



![image-20210828155513241](https://gitee.com/lin2000h/images/raw/master/image-20210828155513241.png)

![image-20210828155520537](https://gitee.com/lin2000h/images/raw/master/image-20210828155520537.png)



### topk()

```python
# 
torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)
```

- **input**：一个tensor数据
- **k**：指明是得到前k个数据以及其index
- **dim**： 指定在哪个维度上排序， 默认是最后一个维度
- **largest**：如果为True，按照大到小排序； 如果为False，按照小到大排序
- **sorted**：返回的结果按照顺序返回
- **out**：可缺省，不要





### seed生成固定随机数

```python
# 为CPU设置种子用于生成随机数，以使得结果是确定的
torch.manual_seed(seed) 

#为当前GPU设置随机种子；
torch.cuda.manual_seed(seed)
# 如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
torch.cuda.manual_seed_all(seed)

# 用于为 Python 中的伪随机数生成器算法设置种子
np.random.seed(seed)
# 设置种子 seed
random.seed(seed)


# 如果配合上设置 Torch 的随机种子为固定值的话，
# 应该可以保证每次运行网络的时候相同输入的输出是固定的
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```



### math.ceil()和math.floor()

```python
# 对一个数进行上取整
math.ceil()
# 对一个数进行下取整
math.floor()
```



### np.concatenate()

```python
# 拼接函数 默认第0维进行拼接
np.concatenate()
```

### np.dot()

**np.dot()**函数主要有两个功能，**向量点积**和**矩阵乘法**



**np.linalg.inv()**：**矩阵求逆**



**np.abs(x)**、**np.fabs(x)** ： **计算数组各元素的绝对值**





### 关于禁止运行脚本

无法加载文件 **C:\Users\Vanous\Documents\WindowsPowerShell\profile.ps1**，因为在 此系统上禁止运行脚本。有关详细信息，请参阅 https:/go.microsoft.com/fwlink/?LinkI D=135170 中的 about_Execution_Policies。 所在位置 行:1 字符: 3

```cmd
# 打开powershell  查看自己当前的执行策略是否为  Restricted
get-executionpolicy

# 以管理员身份打开powershell  将执行策略修改为  RemoteSigned
set-executionpolicy remotesigned
```

**Restricted** 执行策略不允许任何脚本运行。 

**AllSigned 和 RemoteSigned** 执行策略可防止 **Windows PowerShell** 运行没有数字签名的脚本。

 本主题说明如何运行所选未签名脚本（即使在执行策略为 **RemoteSigned** 的情况下），还说明如何对 脚本进行签名以便您自己使用。





### torch.stack()和torch.cat()

一般**torch.cat()**是为了把函数**torch.stack()**得到tensor进行拼接而存在的。

**torch.cat()** 和python中的内置函数**cat()**， 在使用和目的上，是没有区别的，区别在于前者操作对象是tensor。

```python
# 沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状。
outputs = torch.stack(inputs, dim=0) → Tensor

# 函数目的： 在给定维度上对输入的张量序列seq 进行连接操作。
outputs = torch.cat(inputs, dim=0) → Tensor

# inputs : 待连接的张量序列，可以是任意相同Tensor类型的python 序列
# dim : 选择的扩维, 必须在0到len(inputs[0])之间，沿着此维连接张量序列。
```





### scatter_()

[scatter参考链接]: https://blog.csdn.net/guofei_fly/article/details/104308528
[scatter参考链接]: https://www.freesion.com/article/91341387111/

**scatter()** 和 **scatter_()** 的作用是一样的，只不过 **scatter()** 不会直接修改原来的 Tensor，而 **scatter_()** 会

`torch.Tensor.scatter_()`是`torch.gather()`函数的方向反向操作。两个函数可以看成一对兄弟函数。[`gather`](https://blog.csdn.net/wangkaidehao/article/details/104349011)用来解码one hot，`scatter_`用来编码one hot。



PyTorch 中，一般函数加**下划线**代表直接在原来的 **Tensor** 上修改

`scatter_`(*dim*, *index*, *src*) → Tensor

- **dim：**沿着哪个维度进行索引

- **index：**用来 **scatter** 的元素索引

- **src：**用来 **scatter** 的源元素，可以是一个标量或一个张量

  

```python
torch.manual_seed(2333)
src = torch.rand([2,5])
print(src)
# target = torch.zeros(3,5)
# print(target)
target = torch.zeros(3,5).scatter_(0,torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]),src)
print(target)
```

![image-20210828155538979](https://gitee.com/lin2000h/images/raw/master/image-20210828155538979.png)

![image-20210828155545513](https://gitee.com/lin2000h/images/raw/master/image-20210828155545513.png)



### np.clip()

**clip()函数**将数组中的元素限制在**a_min, a_max**之间，大于**a_max**的就使得它等于 **a_max**，小于**a_min**,的就使得它等于**a_min**

**numpy.clip(a, a_min, a_max, out=None)[source]**

```python
x=np.array([[1,2,3,5,6,7,8,9],[1,2,3,5,6,7,8,9]])
np.clip(x,3,8)

Out[90]:
array([[3, 3, 3, 5, 6, 7, 8, 8],
       [3, 3, 3, 5, 6, 7, 8, 8]])
```







### round()

**round()** 方法返回浮点数x的**四舍五入**值。

- x -- 数值表达式。
- n -- 数值表达式，表示从小数点位数。

```python
round( x [, n]  )
```



### torch.bmm()

[bmm和matmul参考链接]: https://blog.csdn.net/foneone/article/details/103876519

**torch.bmm()**强制规定维度和大小相同

**torch.matmul()**没有强制规定维度和大小，可以用利用广播机制进行不同维度的相乘操作

==当进行操作的两个tensor都是3D时，两者等同==



`torch.``bmm`(*input*, *mat2*, *out=None*) → Tensor

**torch.bmm()**是tensor中的一个相乘操作，类似于矩阵中的**A*B**。

参数：

**input，mat2**：两个要进行相乘的tensor结构，**两者必须是3D维度**的，每个维度中的大小是相同的。

**output**：输出结果

并且相乘的两个矩阵，要满足一定的维度要求：input（p,m,**n**) * mat2(p,**n**,a) ->output(p,m,a)。这个要求，可以类比于矩阵相乘。前一个矩阵的列等于后面矩阵的行才可以相乘。





### div

div函数就是对数据进行**标准化**



### random.choice()

**random.choice()**函数：从给定的1维数组中随机采样的函数。

参数
		**numpy.random.choice(a, size=None, replace=True, p=None)**

**a** : 如果是一维数组，就表示从这个一维数组中随机采样；如果是int型，就表示从0到a-1这个序列中随机采样。

**size** : 采样结果的数量，默认为1.可以是整数，表示要采样的数量；也可以为tuple，如(m, n, k)，则要采样的数量为m * n * k，size为(m, n, k)。

**replace** : boolean型，replace指定为True时，采样的元素会有重复；当replace指定为False时，采样不会重复。

**p** : 一个一维数组，制定了a中每个元素采样的概率，若为默认的None，则a中每个元素被采样的概率相同。



### normalize()

`torch.nn.functional.``normalize`(*input*, *p=2*, *dim=1*, *eps=1e-12*, *out=None*)

本质上就是按照某个维度计算范数，**p**表示计算**p范数**（默认为2范数），**dim**计算范数的**维度**（这里为1，一般就是通道数那个维度）



### numpy.std()

计算矩阵的**标准差**

```python
a = np.array([[1, 2], [3, 4]])
np.std(a) # 计算全局标准差
>>> 1.1180339887498949

np.std(a, axis=0) # axis=0计算每一列的标准差
>>> array([ 1.,  1.])

np.std(a, axis=1) # 计算每一行的标准差
>>> array([ 0.5,  0.5])
```



### model.train()和model.eval()

如果模型中有**BN层(Batch Normalization）**和**Dropout**，需要在训练时添加**model.train()**，在测试时添加**model.eval()**。其中**model.train()**是保证**BN**层用**每一批数据的均值和方差**，而**model.eval()**是保证**BN**用**全部训练数据的均值和方差**；而对于**Dropout**，**model.train()**是**随机**取**一部分网络**连接来训练更新参数，而**model.eval()**是利用到了**所有**网络连接。

