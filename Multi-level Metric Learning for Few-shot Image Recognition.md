# Multi-level Metric Learning for Few-shot Image Recognition

# 用于小样本图片识别的多级度量学习

本文采用基于**度量学习**的方式实现小样本学习任务，是在**DN4**上的改进版。作者首先指出当前方法的弊端：

1.忽略了**query image**的分布，理应设计一个**distribution-level similarity**度量**query images**和**support set**之间**分布**的相似度（这里其实应该是计算的一个**分布差异**）

2.仅仅计算一个**pixel-level similarity**是不够全面的（基于**DN4**），应该采用**multi-level similarity metric**来生成更多具有判别性的特征，有益于泛化。

基于以上弊端，作者提出了**pixel-level similarity**、**part-level similarity**和d**istribution-level similarity**，以**三个**不同层次的相似性度量进行分类；这样**类内**的**support images**能够更紧密的分布在更小的特征空间中，从而生成更具有代表性的特征映射。



### **Pixel-level** Similarity Metric

这部分直接参考**Revisiting Local Descriptor based Image-to-Class Measure for Few-shot Learning**，一模一样。

![image-20210828155810226](https://gitee.com/lin2000h/images/raw/master/image-20210828155810226.png)

### **Part-level** **Similarity Metric**

这部分其实就是换了个维度，（自己想的形象比喻：**土豆条**换成**土豆片**）

![image-20210828155817739](https://gitee.com/lin2000h/images/raw/master/image-20210828155817739.png)

**X**: an image     $\mathcal{F}$ 特征提取器**embedding**

$$
\mathcal{L}^{part}=\mathcal{F}_{\theta}(X) = [x_1,x_2,...,x_C] \in R^{HW \times C}
$$

**query image：**
$$
\mathcal{L}^{part}_{Q}=\left[ u_1^{part},u_2^{part},...,u_C^{part} \right] \in \mathbb{R}^{HW\times C}
$$

**support set：**
$$
\mathcal{L}^{part}_{S}=\left[ v_1^{part},v_2^{part},...,v_{MC}^{part} \right] \in \mathbb{R}^{HW\times MC}
$$
**求余弦相似度：**
$$
R_{i,j}^{part}=cos\left( u_{i}^{part},v_{j}^{part} \right)
$$






### **Distribution-level** **similarity metric**

The distribution of pixel-level feature descriptors are multivariate Gaussian（分布服从**高斯分布**）
$$
\mathcal{Q}\,\,=\,\,\mathcal{N}\left( \mu _Q,\varSigma _Q \right)
$$

$$
\mathcal{S}\,\,=\,\,\mathcal{N}\left( \mu _S,\varSigma _S \right)
$$

$\mathcal{F}$​  为度量距离
$$
D_{dis}(Q,S)=\mathcal{F}(Q,S)
$$




#### Kullback-Leibler divergence：

$$
\mathcal{F_{KL}}(S||Q) = \frac {1}{2}(trace(\varSigma_Q^{-1} \varSigma_S)+ ln(\frac{det \varSigma_Q}{det \varSigma_S}) \\+ 
({\mu_Q - \mu_S})^T \varSigma_Q^{-1}(\mu_Q - \mu_S)-c)
$$



#### Wasserstein distance：

$$
\mathcal{F_{Wass}}(Q,S) = ||\mu_Q - \mu_S||_2^2 + trace(\varSigma_Q + \varSigma_S - 
2(\varSigma_Q^{\frac {1}{2}} \varSigma_S \varSigma_Q^{\frac {1}{2}} )^{\frac {1}{2}})  \\
$$

$$
\mathcal{F_{Wass}}(Q,S) = ||\mu_Q - \mu_S||_2^2 + ||\varSigma_Q - \varSigma_S||_F^2
$$





==Wasserstein距离相比KL散度和JS散度的**优势**在于：即使两个分布的支撑集没有重叠或者重叠非常少，仍然能反映两个分布的远近。而JS散度在此情况下是常量，KL散度可能无意义。==



### **Fusion Layer**

$$
\mathcal{D}\left( \mathcal{Q},S \right) = w_1 \cdot \mathcal{D}_{part}\left( Q,S \right) + w_2 \cdot \mathcal{D}_{pixel}\left( Q,S \right) - w_3 \cdot \mathcal{D}_{dis}\left( Q,S \right)
$$



### 代码调试

#### 2021-8-11

这个代码很久之前就跑了，写一点印象比较深的；首先两个分布差异的度量没看懂，因为涉及到比较复杂的数学推导；然后代码部分只跑了5-way 1-shot部分的，其实5-way 5-shot也跑了，只不过计算量比较大，所需要的时间比较长；最后就是代码部分的问题，没有公布backone为resnet12的代码，只公布了conv64部分的代码，所以最高准确率有待验证。



