

# LEARNING TO **PROPAGATE LABELS**:**TRANSDUCTIVE **PROPAGATION NETWORK FOR FEW-SHOT LEARNING

# 直推和标签传播

本文提出了一种采用**直推式传播网络**（**Transductive Propagation Network，TPN**）的小样本学习算法。

- **Inductive Learning（归纳学习）**：训练集有真实标签，而测试集没有真实标签，并且训练集和测试集之间不存在重合。

- **Transductive Learning（直推式学习）**：将带有标签的训练集和不带标签的测试集都输入到网络进行训练，然后再预测这部分测试集的结果。==注意：此时的训练集和测试集来自于同一个数据集==

  （**例子**：布置课后作业时，给考试原题，不给答案）

- **Semi-supervised Learning（半监督学习）**：半监督学习中所使用的无标签数据集合测试集是不相同的两个数据集。





![image-20210828155746443](https://gitee.com/lin2000h/images/raw/master/image-20210828155746443.png)

### PROBLEM DEFINITION(问题定义)

​     $\mathcal{c}_{train}$​​​ : **base classes**                                         $\mathcal{c}_{test}$​​​​​ : **novel classes**

每个**episode**： 从 $\mathcal{c}_{train}$​​ 随机取样 **N** 个 **classes**，每个 **classes** 里面选取 **K** 个样本，作为 **support set**；

**support set**           $S= \{(x_1,y_1),(x_2,y_2),...,(x_{N \times K }),(y_{N \times K})\}$​

**query set**             $Q= \{(x_1^*,y_1^*),(x_2^*,y_2^*),...,(x_T^*),(y_T^*)\}$​



### 1.FEATURE EMBEDDING（特征提取）

​		输入 **input** $x_i$        						特征提取网络  **embedding**  $f_{\varphi}$        

​							 $x_i$   --->   $f_{\varphi} (x_i;\varphi)$  



### 2.GRAPH CONSTRUCTION(构建图模型)

利用一个卷积神经网络  $g_{\phi}$​   来**构建图模型**，构建图模型就是将每个样本当作一个**节点**，计算**每个节点**之间的权重 $W_{i.j}$​​ 的过程；常见的计算方式为**高斯相似性函数**：
$$
W_{i,j} = \exp (- \frac {d(x_i,x_j)}{2 \sigma^2})
$$
**d(,)**为**距离度量函数**（这里选取的为**欧式**距离）；$\sigma$​ 表示**长度范围参数**（对模型**影响很大**）



#### Example-wise length-scale parameter

这里的 $\sigma$（长度范围参数），本文选择利用一个 CNN  $g_{\phi}$，为每个样本都学习一个特有的 $\sigma$ 参数 ； $ \sigma _i = g(f(x_i))$  （这里 $x_i \in S \cup Q $ ,也就是在**S**和**Q**的并集上构造**图模型**）, 然后得到两个节点之间的权重；
$$
W_{i,j} = \exp \left( -\frac {1}{2} d ( \frac{f_{\varphi}(x_i)}{{\sigma}_i},\frac{f_{\varphi}(x_j)}{{\sigma}_j}) \right)  \\
			 W \in R^{(N \times K + T)\times (N \times K + T)}
$$
![image-20210828155756539](https://gitee.com/lin2000h/images/raw/master/image-20210828155756539.png)

作者这里还利用规范化的图拉普拉斯算子处理节点之间的权重 W ，得到 $ S=D^{-\frac{1}{2}}WD^{-\frac{1}{2}}$​​​​ ；这里的 **D** 是一个**对角矩阵**，且矩阵 **D** 中 **(i,i)** 处的值为 **W** 矩阵中第 **i** 行的值之和。



### 3.LABEL PROPAGATION（标签传播）

得到了图模型之后，再利用**有标签**的支持集样本来推测**未带标签**的查询集样本，这一过程被称为**标签传播**；先构建一个非负矩阵  $\mathcal{F} \in R^{(N \times K + T) \times N}$​​​   ( **N**: 类别数；**K**：每个类别的支持集样本数 ；**T**：所有查询集样本数 )；

标签矩阵 $Y \in \mathcal{F}$ , $Y_{i,j}$ 表示第 i 个样本，属于第 j 个类别的概率；矩阵 F 通过迭代训练的方式进行更新，迭代过程如下：
$$
F_{t+1} = \alpha S F_t + (1-\alpha)Y
$$
$F_t$  表示第 **t** 次迭代的预测矩阵；$Y \in \mathcal{F}$​ ，  $Y_{i,j}=1$  表示第 **i** 个样本属于支持集，且其标签为 **j** ；否则为 $Y_{i,j} = 0$​  ；**S** 为**规范化**的权重矩阵；$\alpha \in (0,1) $​​为 控制传播信息量的参数；经过迭代后，F 收敛为 
$$
F^* =(I-\alpha S)^{-1} Y
$$


###  CLASSIFICATION LOSS GENERATION （损失计算）

利用 **softmax函数** 将最后收敛的预测矩阵 $F^*$​ 转化为概率 ，$ \tilde{y_i}$​  为最后的**预测标签**  $y_i$ 为**真实标签**
$$
P(\tilde{y_i}=j|x_i) = \frac{\exp(F_{i,j}^*)}{\sum_{j=1}^N\exp(F_{i,j}^*)}
$$
损失函数：
$$
\mathbf{J}(\varphi,\phi)=\sum_{i=1}^{N \times K + T}\sum_{j=1}^N - \mathbb{I}(y_i === j) \log(P(\tilde{y_i}=j|x_i))
$$


这里的  $\mathbb{I}(b)$  是一个**指标函数**，当 b 为 **true**，$\mathbb{I}(b) = 1$ ，否则 $\mathbb{I}(b) = 0$



### 个人总结

本文代码并未跑，所以代码具体是怎么实现标签传播的不清楚，因为自己在看论文时，标签传播这一块没看懂。
