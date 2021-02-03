## 泛化

- 什么是泛化
- 泛化在概率上的可能性
- 什么是 vc dimension
- 模型的复杂度
- 什么是 generalization bound, 与哪些因素有关

  
## overfitting. regularization, validation.

what is overfitting
模型的 capacity
the role of noise
bias-variance-noise 分解


## principles: Occam's razor, data snooping, sampling bias



# 机器学习中的 bias 和 variance

bias-variance 分解是机器学习中非常重要的一个工具. 通过把 E_out 分解为 bias 和 variance 两部分, 有助于

$\bar{g}$ 的概念. data generating process, data generating distribution.

定义: bias 是 $\bar{g}$ 与 f 的差距, variance 是 $\bar{g}$ 与 g 的差距

MSE 分解为 bias + variance.

在学习曲线上的表现 (E_out, E_in vs. N). simple model, complex model.

"参数估计" 中的 bias 和 variance

bias variance noise 分解

? overfitting 的发生: noise 诱使模型跑偏, 造成很大 variance

西瓜书 P46: 1. bias, variance 的含义, 2. 随训练程度的变化, bias variance 比重的变化


主要参考: 
- LFD lec8
  - 进一步, 是否提一下 bias-variance-noise 分解?
- cs229 ML-advice
- dlbook ch5
- 西瓜书 ch2.5
