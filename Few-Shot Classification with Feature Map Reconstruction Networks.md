# Few-Shot Classification with **Feature Map Reconstruction** Networks

# 基于**特征图重构**的小样本分类

### 度量学习中的一个问题

**特征提取器**生成的是图片的**feature map**，距离**度量函数**是需要的是**整张图片**的**向量表示**，所以需要将**feature map**转换为**向量表示**；在**理想状态**下，转换为**向量表示**后，可以保留**feature map**的**粒度信息**和**细节信息**，同时模型也**不**会过拟合；现存方法有：

- **全局平均池化**：也就是粗暴的抛弃一些空间上的信息
- 打平整张**feature map**，变成一个**长向量**



本文就是为了更好的将**feature map**转换为**向量表示**，同时也能保留**空间**上的**细节**，提出来**FRN**（**Feature Map Reconstruction Networks**）



### Feature Map Ridge Regression

$x_s$:   **support images**                 $x_q$:  **query images**

$x_q$    经过**embedding**之后    $Q \in R^{r \times d}$     这里的（**r = h $\times$​​ w    h**和**w**分别代表图片的**高**和**宽**）



对于每个类别 **c** $\in$​​​​​  **C**，当前**类别** **c** 中所有的 **k** **张图片 **；用当前**类 c** 的**所有图片**来构建一个**矩阵**

 $S_c \in R^{ kr \times d}$​​      找到一个**矩阵** $W \in  R^{r \times kr}$​​     能够使得 $WS_c \thickapprox Q$​​

 

当前问题转换为一个**求最优矩阵  $\bar {W}$​  的最小二乘问题**：
$$
\bar{W} = \arg \min_W ||Q-WS_c||^2 + \lambda||W||^2
$$
对 W 进行求导， 则当 求导出来的式子为 0 时，
