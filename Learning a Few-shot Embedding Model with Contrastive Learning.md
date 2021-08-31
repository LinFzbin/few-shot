



# Learning a Few-shot Embedding Model with Contrastive Learning

# 基于对比学习的小样本嵌入模型

​																														**AAAI 2021 腾讯优图实验室**



​																==**对比学习+数据增强**==

### Contrastive Learning

拉近相似样本之间的距离，推远不相似样本之间的距离。（**拉近正样本**，**拉远负样本**）

#### 有监督对比学习

将监督样本中**相同label**的样本作为**正样本**，**不同label**的样本作为**负样本**，进行对比学习。



#### 无监督对比学习

**没有label**，同一个样本**构造两个view**（具体地说，就是一张图像经过随机变换，比如裁剪、随机旋转、色彩变化等，一张图像就可以获得几个不同的变体），**同**一个样本构造的**两个view**互为**正样本**；而其他样本构造的view全部为**负样本**。同一个样本构造两个view，属于对数据进行扩增，也可以被称作为数据扩展对比学习。

### motivation

1. 人可以通过图像中目标对象的一部分对其进行识别；因而作者提出通过**对比学习构造正负样本**。
2. 源类的**归纳偏置**，可能会引入**实例**与**类**之间的相关性的**噪声**（例如：有马的图像中通常都有草，这样的图像训练的模型在对另外的马图像的分类时，可能会倾向于那些视觉上有马也有草的图像）；本文提出通过**混合不同的补丁（PatchMix）**来减轻此问题，强制网络模型学习更多的分离信息。



### Training Phase

为**query**构造对比对（对于每个**query instance**  $x_i^q$​​​​​  ，都有其**label **  $y_i^q$​​​​​​  ），将相同**label**的**support**作为**正样本**，不同**label**的**support**作为**负样本**；一个**query instance** 的**infoNCE**为：


$$
L_i = - \log \frac{\sum_{y_j^s = y_i^q \exp^{f_i^q \cdot f_j^s}}}{\sum_{y_j^s = y_i^q \exp^{f_i^q \cdot f_j^s}} + \
\sum_{y_k^s \neq y_i^q \exp^{f_i^q \cdot f_j^s}}}
$$
对于一个**episode**，整体损失是所有**query instance**的**infoNCE**均值；**supervised loss**的权重为**1.0**，**infoNCE loss**的权重为**0.5**。

### Testing Phase

测试阶段是有**label**的**support sample**和**无label**的**query sample**，对于每个**query sample** ,找到内积最大的**support sample**，那么此**query sample**的预测标签就为找到的内积最大的support sample的标签( $\hat{y}i^q  =  y{j*}s$​​ )。

​	
$$
j^* = \arg \max_j f_i^q f_j^s
$$




![image-20210828155648018](https://gitee.com/lin2000h/images/raw/master/image-20210828155648018.png)

### Construct Hard Samples

​																	==只用于**training**阶段==



对**support images**进行**random block**（也就是对**support images**进行**random masks**（随机掩码），这样**random masks**之后的**support images**比原图更加难以识别）。



将**query image**切成多个**patches**（图中所示为**2*2**，也就是切成了**4**个**patches**），这样做是希望切割出来的每个**patch**也能被正确分类（也就是**query image**的一部分图像也能被正确的分类）


$$
L_{iwh} = - \log \frac{\sum_{y_j^s = y_i^q \exp^{f_{iwh}^q \cdot f_j^s}}}{\sum_{y_j^s = y_i^q \exp^{f_{iwh}^q \cdot f_j^s}} + \
\sum_{y_k^s \neq y_i^q \exp^{f_{iwh}^q \cdot f_j^s}}}
$$



### Enhancing Contrastive Learning via PatchMix

为了避免引入不必要的噪声，只对**query sample** 做 **PatchMix**；对一个 **query instance**  $x_i^q$​  在当前 **episode **里随机选取一个不同的 **query instance** $x_k^q$​​​​​ ，做上图最右侧的变换；这里每个 **patch** 都有自己的**label** ，与上面一样作分类。





### 论文准确率

![image-20210828155658195](https://gitee.com/lin2000h/images/raw/master/image-20210828155658195.png)

![image-20210828155704863](https://gitee.com/lin2000h/images/raw/master/image-20210828155704863.png)

![image-20210828155716962](https://gitee.com/lin2000h/images/raw/master/image-20210828155716962.png)

![image-20210828155727865](https://gitee.com/lin2000h/images/raw/master/image-20210828155727865.png)

准确率如上，准确率完全复现原文的；在epoch=60及以后调整学习率之后，可以看到准确率慢慢有所提高；epoch=90（总的epoch=90）的时候准确率是最高的。

### 个人总结

#### 2021-8-18

这篇文章大概是看完了，也大概明白文章主要的做法，但是因为连不上服务器，代码跑不了，不能去验证论文的创新点。



### 代码调试

#### 21-8-22

代码调试成功，再进一步对照论文看代码

#### 21-8-25

- **PatchMix（query images做此操作）：**先将每个**batch**（这里每个**batch**的**episode=4**）里面的**30**张图片打乱；再随机选取一个框[**rand_bbox()**函数来操作]；最后进行**PatchMix**，每个**episode**的**PatchMix**的框都是一样的。



- **RandomBlock：**这部分其实也蛮简单，也就是根据最后的图片尺寸大小**[11,11]**生成一个遮罩**mask**，随机生成一个坐标**（x,y）**，此处**(x,y)**的坐标置为**0**，其余的置为**1**，和原始特征图相乘，**(x,y)**处的图像就相当于被遮罩住了。



今天算把这篇论文的代码看的差不多了，准确率还在跑，**epoch=20**的准确率已经达到**58.02%**；然后求的两个损失那部分需要再仔细看下代码，对照论文好好理解下论文所说的如何构造**对比对**。

#### 21-8-26

如何构造对比对这部分代码，现在还是懵的，看不太懂，以后有时间继续看吧，最近又发现两篇比较有意思的论文，准备接着看。
