# Revisiting Local Descriptor based Image-to-Class Measure for Few-shot Learning



本文采用基于**度量学习**的方式实现小样本学习任务，与其他度量学习方式不同；本文所用的**embedding**特征提取器**只有卷积层**，**没有全连接层**；也就是没有将一幅图像的特征信息压缩到一个紧凑的**图像级别**的特征表示（特征提取网络将图像特征高度**抽象化**，对**位置不敏感**的向量），因为作者认为图像级别的特征信息会损失很多有**区分性**的特征。本文注重于图像的**局部特征**表示，在**support set**所有的**local descriptors**中找出与**query image**中每个**local descriptor**最相似的**k**个**local descriptors**，这样每个**query image**的每个**local descriptor**对于**support set**的每个类别**c**都有一个相似度分数，进而每张**query image**对于**support set**的每个类别**c**也有一个相似度分数，分数最高的即为正确分类。



### DN4网络

全称 **Deep Nearest Neighbor Neural Network**，**简写为 DN4**

**embedding module $\psi$​**：==只包含卷积层，没有全连接层== （因为要用到图片中的局部信息，所以不将图片的特征向量打平）

**image：X**
$$
\psi(X)=[x_1,x_2,...,x_m] \in R^{d\times m}  \quad m=h\times w
$$
这里的 $x_i$ ​就是论文所提出的**local descriptor**（局部描述子）

将一张图片先随机裁剪及一些操作后，得到图片的尺寸为**84*84**(h=84,w=84)； 再经过**embedding**特征提取模块后，这里的图片尺寸变为了**21*21**(h=21,w=21);这里将一个**local descriptor**看做成一张图片的一部分局部信息（因为**local descriptor**虽然是一个像素点，但是图片是经过缩放的，所以可以看成原图的一个局部块，也就是原图中的一部分局部信息）。



### Image-to-Class module 

$\Phi$​​​​ 用一个类中的所有图片的**local descripto**r构成该类的局部描述特征空间，通过**KNN**来计算图像到类之间的相似性（距离）



![Flowchart](file://E:\typora\few-shot%20study\Revisiting%20Local%20Descriptor%20based%20Image-to-Class%20Measure%20for%20Few-shot%20Learning.assets\Flowchart.bmp?lastModify=1630137503)



对于一张 **query image**： $\psi(q)= [x_1,x_2,...,x_m] \in R^{d\times m} $

对一张 **query image** 中的每个 **loacl descriptor** 在 **support set** 每个**类别 c** 中找到 **k 近邻**个   $\hat{x_i}|_{j=1}^k$  ,计算 $x_i$ 和 $\hat{x_i}$ 的相似度，相当于一张 **query image** 和 一个**类别 c** 要求 **mk** 个相似度 （一张 **query image** 有 **m** 个 **local descriptor** ，每个**local descriptor** 在当前**类别 c** 中找到 **k** 近邻个   $\hat{x_i}$​​​​​ ）

下面为求相似度的公式（距离度量为**余弦距离**）


$$
cos(x_i,\hat{x_i}^j) = \frac{x_i^T \cdot x_i^T}{||x_i||\cdot ||\hat{x_i}||}    \\

\begin{aligned}
	\Phi(\psi(q),c) = \sum_{i=1}^m \sum_{j=1}^k cos(x_i,\hat{x_i}^j)  \\
	 = \sum_{i=1}^m \sum_{j=1}^k \frac{x_i^T \cdot x_i^T}{||x_i||\cdot ||\hat{x_i}||}
 \end{aligned}   \\
$$



对于一张**query image**的每个**local descriptor**，都找到每个类别**c**中最近邻的**k**个**local descriptors**；然后一张**query image**对每个类别**c**都有一个相似度，取相似度最大的类别为正确分类。

### 代码调试

#### 2021-8-10

昨天将代码调通看了下准确率（这里直接调的是**ResNet256F 5-ways 1-shot**），当然和论文所展示的准确率有所占据，在1%左右。然后关注到一个比较重要的点是：这篇论文是从头开始端到端的训练的，没有预训练的过程（也就是没有在**miniimagenet**数据集上的**64classes**进行监督学习）。



#### 2021-8-11

这里记录一下实验的episode相关设置：

​	总 **epoch=30** ，episode_**train**_num=**10000**，每个**episode**随机取样一个**N-ways K-shots**任务；episode_**val**_num =**1000**，每个episode随机取样一个**N-ways K-shots**任务；episode_**test**_num =**1000**，每个episode随机取样一个**N-ways K-shots**任务。

embedding ：[3,84,84] --> [64,21,21]

