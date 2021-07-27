------

# Learning Intact Features by Erasing-Inpainting for Few-shot Classification 

# 						基于擦除修复学习完整特征的小样本分类

本文提出了**Cross-Set Erasing-Inpainting（CSEI）**，**CSEI**模块由**erasing-inpainting**（擦除修复）和**cross-set data augmentation**（跨集数据增广）组成。在训练阶段，先擦除掉support set中图像中具有辨识度的区域，然后用image inpainting来补全图片，将处理后的图片加入到对应的query set中一起进行训练。本文还提出了**task-specific feature modulation（TSFM）**模块，这个模块由空间注意力和通道注意力组成，使得feature能够很好的适应到当前的任务中。



**N-ways M-shots**



​		$S^{train}=\{S^c\}^N_{c=1}$                $|S^c|=M$                  $s_j$表示$S^{train}$中的样本

​		$Q^{train}=\{Q^c\}^N_{c=1}$               $|Q^c|=M'$               $q_i$表示$Q^{train}$中的样本

​		g表示embedding function

​		$g_{s_j}$和$g_{q_i}$ 分别表示提取出来的feature



**Training**

​		

​		每个类别c的原型向量prototype vector：

​							
$$
m_c=\frac{1}{|S^c|}\sum_{s_j\in{S^c}}g_{s_j}  	 ,	c=1,2,...,N
$$
​		距离函数d，计算一个query sample在类别上的分布：

​							
$$
P_{i,c}=\frac{exp(d(g_{q_i},m_c))}{\sum_c exp(d(g_{q_i},m_c))}
$$



​		标签函数：
$$
I(q_i=c) , \forall{q_i\in{Q^c}}(c=1,2,...,N)
$$




​		损失：
$$
L=-\frac{1}{NM'}\sum_i\sum_c1[I(q_i=c)]logP_{i,c}
$$


**Testing**

​		训练完成后，





























