---
layout: post
title: 极大似然估计与最大后验概率估计
subtitle: 
tags: MachineLearning
category: tech
---



极大似然估计和最大后验概率估计是机器学习中常见的两个基本概念, 尤其是前者. 但一直以来我对这两个概念有很多疑问, 比如:

* 为什么按频次估计概率就是极大似然估计?
* 在朴素贝叶斯法的参数估计中, 为什么不加平滑的估计是极大似然估计, 而平滑化的估计是最大后验概率估计? (这也是《统计学习方法》第四章的两道课后习题.)

直到读了 Tom Mitchell 《机器学习》 新版第二章(草稿) "Estimating Probabilities: MLE and MAP" [^1], 这些疑问才得到解答. 以下是阅读笔记和思考.



### 由抛硬币的例子说起

简单的例子可以帮助我们理解抽象的概念. 那么就来看看"抛硬币"这个经典例子吧. 

假设你有一个硬币, 掷硬币的结果用随机变量 X 表示, X = 1 和 0 分别代表正面和反面. 假设得到正面的真实概率是 $ \theta$ (这个值是未知的, 是我们需要估计的目标). 掷了 n 次以后, 得到正反面的次数分别为 $\alpha_1$ 和 $\alpha_0$. 

现在根据观测结果 $\alpha_1$ 和 $\alpha_0$ 来估计 $\theta$. 一个非常简单直观的估计必然会是:

$$\hat{\theta} = \frac{\alpha_1}{\alpha_1 + \alpha_0} \qquad(1)$$

这其实就是**极大似然估计**. 所谓"极大似然", 指的是这一估计使得观测数据概率最大化. 至于为何如此, 本文后面会给出证明.

根据大数定理, 极大似然估计当数据量充足时是非常可靠的. 但当数据量有限时, 这个估计可能就不太靠谱了. 

这个时候, 如果能结合我们对 $\theta$ 已有的先验知识, 可以提高估计的可靠性, 比如下面这种估计:

$$\hat{\theta} = \frac{\alpha_1 + \gamma_1}{(\alpha_1 + \gamma_1) + (\alpha_0 + \gamma_0)} \qquad(2)$$

这刚好就是**最大后验概率估计**. 其中, $\gamma_1$ 和 $\gamma_0$ 是由先验知识引入的. 对于掷硬币这件事, 我们可以比较肯定地判断, 概率 $\theta$ 在 0.5 附近, 于是, 在观测数据 $\alpha_0$, $\alpha_1$ 的基础上, 我们可以在想象中补充进行一批试验, 其中正反面的次数分别为 $\gamma_1$, $\gamma_0$, 基于先验知识, 我们取 $\gamma_1 = \gamma_0$.

上面第二种估计有以下特点:

* 不仅结合了先验知识, 而且还可以表达对先验知识的确定程度. 比如, 取$\gamma_1 = \gamma_0 = 100$, 相比于取 $\gamma_1 = \gamma_0 = 10$, 代表我们对 "$\theta$ 在0.5附近"这一先验知识的确定性更大.
* 如果取 $\gamma_1 = \gamma_0 = 0$, 这一估计等同于极大似然估计. 即, 后者是前者的一种特例.
* 随着观测数据量的增长, 先验假定对估计结果的影响逐渐衰减.

到这里, 我们通过掷硬币的例子引出了极大似然估计和最大后验概率估计的概念, 接下来, 我们进一步来探讨这两个概念.



### 极大似然估计 (MLE)

极大似然估计的原则是: 对概率的参数 $\theta$ 的估计应使观测到的训练数据 D 的概率最大. 即:

$$\hat{\theta}_{MLE} = \mathrm{argmax}_\theta P(D\vert\theta)$$

其中, $P(D\vert\theta)$ 当看做 $\theta$ 的函数时, 称为似然函数.

**下面由极大似然估计的定义推出式(1):**

首先, 似然函数为:

$$L(\theta) = P(D\vert\theta) = \theta^{\alpha_1}(1-\theta)^{\alpha_0} \qquad(3)$$

为便于处理, 取对数:

$$\ln P(D\vert\theta) = \alpha_1 \ln \theta + \alpha_0 \ln (1-\theta)$$

极大化 $P(D\vert\theta)$ 等价于极大化 $\ln P(D\vert\theta)$. 而极值点处有 $\frac{\partial \ln P(D\vert\theta)}{\partial \theta} = 0$

求偏导 (过程略) 可得: $\theta = \frac{\alpha_1}{\alpha_1 + \alpha_0}$, 即式(1).







### 最大后验概率估计 (MAP 估计)

参数的最大后验概率估计是指, 给定观测数据 D 和先验假定, 取可能性最大的参数值. 即:

$$\hat{\theta}_{MAP} = \mathrm{argmax}_{\theta} P(\theta\vert D)$$

根据贝叶斯定理, $P(\theta\vert D) = \frac{P(D\vert\theta)P(\theta)}{P(D)}$.

其中分母与$\theta$ 无关, 因此:

$$\hat{\theta}_{MAP} = \mathrm{argmax}_{\theta} P(D\vert\theta) P(\theta)$$

所以 MAP 估计与 MLE 估计的唯一区别是多了一项先验概率 $P(\theta)$.

**下面由最大后验概率估计推出式(2).**

首先需要指定一个先验分布. 在掷硬币问题中, 观测数据服从伯努利分布, 这种情况下常用的先验分布是 [Beta 分布](https://en.wikipedia.org/wiki/Beta_distribution):

$$P(\theta) = Beta(\beta_0, \beta_1) = \frac{\theta^{\beta_1 - 1} (1-\theta)^{\beta_0 - 1}}{B(\beta_0, \beta_1)}$$

其中, 分母 $B(\beta_0, \beta_1)$ 的作用是归一化, 与 $\theta$ 无关. 因此:

$$\hat{\theta}_{MAP} = \mathrm{argmax}_{\theta} P(D\vert\theta) P(\theta)$$

$$= \mathrm{argmax}_\theta \theta^{\alpha_1}(1-\theta)^{\alpha_0} \frac{\theta^{\beta_1 - 1} (1-\theta)^{\beta_0 - 1}}{B(\beta_0, \beta_1)}$$

$$=\mathrm{argmax}_\theta \theta^{\alpha_1+\beta_1-1} (1-\theta)^{\alpha_0+\beta_0-1}$$

到这里可以发现, 上式在形式上与式(3)的似然函数是一样的! 只需把 $\alpha_1+\beta_1+1$ 和 $\alpha_0+\beta_0-1$ 分别看做整体. 于是, 我们可以直接套用上一节推导的结论, 得到:

$$\hat{\theta}_{MAP} = \frac{\alpha_1+\beta_1-1}{(\alpha_1+\beta_1-1) + (\alpha_0+\beta_0-1)}$$

与式(2)等价.



### 理解朴素贝叶斯的参数估计

以上通过掷硬币的例子, 对 MLE 和 MAP 估计分别给出了一种简单直观的典型形式, 有助于获得一种直觉上的理解. 进一步, 我们就可以回答本文开头对朴素贝叶斯的参数估计的两个疑问.

* 基于频次且不考虑平滑的估计是极大似然估计.

![naive_bayes_mle](https://d2mxuefqeaa7sj.cloudfront.net/s_9CD672DFD4E0CDCF730BC19B22A9E2904CE720B6E37BB0434E3844F2944DD4EB_1504523084769_image.png)

上式本质上与掷硬币例子的式(1)相同.

* 考虑平滑的估计是贝叶斯估计

![naive_bayes_map](https://d2mxuefqeaa7sj.cloudfront.net/s_9CD672DFD4E0CDCF730BC19B22A9E2904CE720B6E37BB0434E3844F2944DD4EB_1504523075268_image.png)

这里平滑项 $\lambda$ 的作用相当于式(2)中的 $\gamma_0, \gamma_1$, 反映了一种均匀的先验假定. $\lambda$ 的取值决定了先验假定影响的强弱.



### 小结





---

[^1]: Tom Mitchell, Machine Learning - Ch2 Estimating Probabilities: MLE and MAP (draft), http://www.cs.cmu.edu/~tom/mlbook/Joint_MLE_MAP.pdf