## **基于数据增强的小样本学习**

小样本学习的根本问题在于样本量过少，从而导致样本多样性变低。在数据量有限的情况下，可以通过数据增强(data augmentation)来提高样本多样性。本文将将基于数据增强的方法分为基于**无标签数据**、基于**数据合成**和基于**特征增强**的方法三种。



### **基于无标签数据的方法：**

利用无标签数据对小样本数据集进行扩充。

#### **半监督学习**

**2016 年,Wang 等人：**Wang YX, Hebert M. Learning from small sample sets by combining unsupervised meta-training with CNNs. In: Advances in Neural Information Processing Systems. 2016. 244−252.
		**2018年，改进MAML+半监督学习：**Boney R, Ilin A. Semi-supervised few-shot learning with MAMLl. In: Proc. of the ICLR (Workshop). 2018.
		**2018年，改进原型网络+无标签数据：**Ren MY, Triantafillou E, Ravi S, et al. Meta-learning for semi-supervised few-shot classification. arXiv preprint arXiv:1803. 00676, 2018.



#### **直推式学习**

直推式学习可看作半监督学习的子问题。直推式学习假设未标注数据是测试数据，目的是在这些未标记数据上取得最佳泛化能力。

**2019年，转导传播网络(transductive propagation network)** Liu Y, Lee J, Park M, et al. Learning to propagate labels: Transductive propagation network for few-shot learning. arXiv preprint arXiv:1805.10002, 2018.
		**2019年，交叉注意力网络：**Hou RB, Chang H, Ma BP, et al. Cross attention network for few-shot classification. In: Advances in Neural Information Processing Systems. 2019. 4003−4014.



### **基于数据合成的方法**

基于数据合成的方法是指为小样本类别合成新的带标签数据来扩充训练数据

> **生成对抗网络（GAN）：**Mehrotra A, Dukkipati A. Generative adversarial residual pairwise networks for one shot learning. arXiv preprint arXiv:1703.08033, 2017.
> 		**表示学习+小样本学习：**在含有大量数据的源数据集上学习通用的表示模型，之后在少量数据新类别中微调模型。Hariharan B, Girshick R. Low-shot visual recognition by shrinking and hallucinating features. In: Proc. of the IEEE Int’l Conf. on Computer Vision. 2017. 3018−3027.
> 		**元学习+数据生成：**通过数据生成模型生成虚拟数据来扩充样本的多样性, 结合元学习方法,通过端到端方法共同训练生成模型和分类算法.Wang YX, Girshick R, Hebert M, et al. Low-shot learning from imaginary data. In: Proc. of the IEEE Conf. on Computer Vision and Pattern Recognition. 2018. 7278−7286.
> 		**变分编码器(VAE)+ GAN：**充分利用了两者的优势集成了一个新的网络 f-VAEGAN-D2. Xian Y, Sharma S, Schiele B, et al. f-VAEGAN-D2: A feature generating framework for any-shot learning. In: Proc. of the IEEE Conf. on Computer Vision and Pattern Recognition. 2019. 10275−10284.
> 		**元学习：**利用元学习对训练集的图像对支持集进行插值,形成扩充的支持集集合 Chen Z, Fu Y, Kim YX, et al. Image deformation meta-networks for one-shot learning. In: Proc. of the IEEE Conf. on Computer Vision and Pattern Recognition. 2019. 8680−8689.

### **基于特征增强的方法**



以上两种方法都是利用辅助数据来增强样本空间,除此之外,还可通过增强样本特征空间来提高样本的多样性,因为小样本学习的一个关键是如何得到一个泛化性好的特征提取器.



> **2017,AGA模型：**学习合成数据的映射,使样本的属性处于期望的值或强度. Dixit M, Kwitt R, Niethammer M, et al. AGA: Attribute guided augmentation. In: Proc. of the IEEE Conf. on Computer Vision and Pattern Recognition. 2017. 7455−7463.
> 		**特征迁移网络(FATTEN)：**用于描述物体姿态变化引起的运动轨迹变化 Liu B, Wang X, Dixit M, et al. Feature space transfer for data augmentation. In: Proc. of the IEEE Conf. on Computer Vision and Pattern Recognition. 2018. 9090−9098.
> 		**Delta 编码器：**通过看到少量样本来为不可见的类别合成新样本,将合成样本 用于训练分类器.该模型既能提取同类训练样本之间可转移的类内变形,也能将这些增量应用到新类别的小样本中,以便有效地合成新类样本. Schwartz E, Karlinsky L, Shtok J, et al. Delta-encoder: An effective sample synthesis method for few-shot object recognition. In: Advances in Neural Information Processing Systems. 2018. 2845−2855.
> 		**双向网络TriNet：**图像的每个类别在语义空间中具有更丰富的特征,所以通过标签语义空间和图像特征空间的相互映射,可以对图像的特征进行增强 Chen Z, Fu Y, Zhang Y, et al. Semantic feature augmentation in few-shot learning. arXiv preprint arXiv:1804.05298, 2018.
> 		**对抗特征：**提出可以把固定的注意力机制换成不确定的注意力 机制 M.输入的图像经提取特征后进行平均池化,分类得到交叉熵损失 l.用 l 对 M 求梯度,得到使 l 最大的更新方向从而更新 M. Shen W, Shi Z, Sun J. Learning from adversarial features for few-shot classification. arXiv preprint arXiv:1903.10225, 2019.



通过梳理基于数据增强的小样本学习模型的研究进展,可以思考未来的两个改进方向.

\1) 更好地利用无标注数据:：由于真实世界中存在着大量的无标注数据,不利用这些数据会损失很多信息,更好、更合理地使用无标注数据,是一个非常重要的改进方向.

\2) 更好地利用辅助特征：小样本学习中,由于样本量过少导致特征多样性降低.为提高特征多样性,可利用辅助数据集或者辅助属性进行特征增强,从而帮助模型更好地提取特征来提升分类的准确率.







### **3.3 度量学习**



通过计算待分类样本和已知分类样本之间的距离,找到邻近类别来确定待分类样本的分类结果。基于度量学习方法的通用流程具有两个模块:嵌入模块和度量模块,将样本通过嵌入模块嵌入向量空间,再根据度量模块给出相似度得分.



> **孪生神经网络**(siamese neural network)**：**孪生神经网络从数据中学习度量,进而利用学习到的度量比较和匹配未知类别的样本,两个孪生神经网络共享一套参数和权重. Koch G, Zemel R, Salakhutdinov R. Siamese neural networks for one-shot image recognition. In: Proc. of the ICML Deep Learning Workshop. 2015
> 		**匹配网络：**Vinyals O, Blundell C, Lillicrap T, et al. Matching networks for one shot learning. In: Advances in Neural Information Processing Systems. 2016. 3630−3638.
> 		**LSTM+匹配网络：**Jiang LB, Zhou XL, Jiang FW, Che L. One-shot learning based on improved matching network. Systems Engineering and Electronics, 2019,41(6):1210−1217
> 		**多注意力网络模型：**Wang P, Liu L, Shen C, et al. Multi-attention network for one shot learning. In: Proc. of the IEEE Conf. on Computer Vision and Pattern Recognition. 2017. 2721−2729.
> 		**原型网络（**prototypical networks**）：**Snell J, Swersky K, Zemel RS. Prototypical networks for few-shot learning. In: Advances in Neural Information Processing Systems. 2017. 4077−4087.
> 2.1中原型网络+无标签数据
> 		**基于人工注意力的原型网络：**Gao TY, Han X, Liu ZY, Sun MS. Hybrid attention-based prototypical networks for noisy few-shot relation classification. In: Proc. of the AAAI Conf. on Artificial Intelligence. 2019. 6407−6414.
> 		**层次注意力原型网络(HAPN)：**Sun SL, Sun QF, Zhou K, Lv TC. Hierarchical attention prototypical networks for few-shot text classification. In: Proc. of the Conf. on Empirical Methods in Natural Language Processing and the 9th Int’l Joint Conf. on Natural Language Processing (EMNLP-IJCNLP). 2019. 476−485.
> 		**关系网络：**Sung F, Yang Y, Zhang L, et al. Learning to compare: Relation network for few-shot learning. In: Proc. of the IEEE Conf. on Computer Vision and Pattern Recognition. 2018. 1199−1208.
> 		**深度比较网络：**Zhang X, Sung F, Qiang Y, et al. Deep comparison: Relation columns for few-shot learning. arXiv preprint arXiv:1811.07100, 2018.
> Hilliard N, Phillips L, Howland S, et al. Few-shot learning with metric-agnostic conditional embeddings. arXiv preprint arXiv:1802.04376, 2018.
> 		**协方差度量网络(CovaMNet)：**Li W, Xu J, Huo J, Wang L, Yang G, Luo J. Distribution consistency based covariance metric networks for few-shot learning. In: Proc. of the AAAI Conf. on Artificial Intelligence. 2019. 8642−8649.
> 		**深度最近邻神经网络（DN4）：**Li W, Wang L, Xu J, Huo J, Gao Y, Luo J. Revisiting local descriptor based image-to-class measure for few-shot learning. In: Proc. of the IEEE Conf. on Computer Vision and Pattern Recognition. 2019. 7260−7268.
> 		Li H, Eigen D, Dodge S , et al. Finding task-relevant features for few-shot learning by category traversal. In: Proc. of the IEEE Conf. on Computer Vision and Pattern Recognition. 2019. 1−10.