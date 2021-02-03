---
layout: post
title: 朴素贝叶斯法用于情感分类
subtitle: 一种简单高效的分类方法
tags: MachineLearning NLP
category: tech
---

### 朴素贝叶斯法

朴素贝叶斯法是一种基于贝叶斯定理的简单高效的分类方法。这种方法引入了一条比较强的假设：在分类确定的条件下，不同特征是相互独立的，因此被冠以「朴素」（naive）之名；通过引入独立假设，可以避免贝叶斯定理求解时面临的组合爆炸、样本稀疏问题。虽然有点「天真幼稚」，但它实际效果还比较好，可以说是性价比很高，甚至还曾入选「数据挖掘十大算法」。[^1]

朴素贝叶斯法的准则是后验概率最大化，其本质是期望风险最小化。具体内容可参看《统计学习方法》第四章，叙述非常简洁。

朴素贝叶斯法若采用极大似然估计，可能会遇到概率值为0的情况 (因为某些特征未在训练集中出现)，影响后验概率的计算。为解决此问题，可采用贝叶斯估计. 贝叶斯估计的一种特殊情形是拉普拉斯平滑(Laplace smoothing, 或称 add-one smoothing)。拉普拉斯平滑在`语言模型`场景下效果不够好, 常用更复杂的平滑算法代替, 但在朴素贝叶斯文本分类场景下则有广泛的应用。[^2]

朴素贝叶斯法在实际应用中，无论对孤立噪声点还是对无关属性，都是一种健壮的分类器[^3]。但如果属性之间有相关性，就容易降低性能。如果想要在模型中考虑属性的相关性，就需要采用更复杂的升级版贝叶斯分类器，比如半朴素贝叶斯、贝叶斯网。


### 朴素贝叶斯法用于文本情感分类

对这类问题，SLP 书中有一章很值得参考[^2]。本节以下均为该书中的要点。

#### 分类器训练

1、在计算条件概率(的极大似然估计)时:

$$ P(w_i|c) = \frac {count(w_i, c)}{ \sum_{w \in V} count(w, c)} $$

需要注意 $\sum_{w \in V} count(w, c) $ 的求和范围是所有分类下的总词表，而不是只是当前分类下的。

2、未知词的处理：遇到未知的词（即训练集中不存在的词）时，这里给出的建议是直接忽略它。另一种做法是，把所有未知词统一标记为‘unknown’，录入词表，并且需要在计算条件概率时考虑^5。稍作分析可知，这两种做法的分类结果会有一些差异，除非不同分类的先验概率相同。

3、stop words: 有时会刻意忽略一些高频词，比如英文中 the、a 这种。确定 stop word list 的方法，一是取最高频的 10-100 个词，二是利用网上现成的 stop word list。但在大多数文本分类应用中，使用 stop word list 并不会提高性能。


#### 训练方法的改进

可尝试用以下方式来改善分类效果：
* 对每一段文本，可忽略不同词出现的具体次数，仅区分出现/未出现。这种方法称为 binary multinominal naive Bayes.
* 处理否定词。一种简单的做法是，把否定词之后、下一个标点符号之前的词 w 都处理为 NOT\_w，比如：`didnt like this movie, but ...` 可处理为 `didnt NOT_like NOT_this NOT_movie, but ...`。
* 训练数据不足时，可以采用现成的情感词汇表（sentiment lexicons）  


#### 分类结果评估

对测试集进行分类预测后，会得到 2x2=4 种结果：[true/false] x [positive/negtive]。它们组成的表称为 contingency table，或混淆矩阵（[confusion matrix](http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/))。

两种常见的评价指标：
* $Precision = tp/(tp+fp)$
* $Recall = tp/(tp+fn)$

实际中常用一个指标 F-measure 来综合考虑以上两项:

$$ F = \frac {(\beta ^2 + 1)PR}{(\beta ^2 P + R)} $$

当 $\beta = 1$ 时，$F\_1 = 2PR/(P+R)$。


### 实例

这里是一个实际例子: [naive_bayes_implementation.ipynb](https://github.com/sunoonlee/machine-learning/blob/master/naive_bayes/naive_bayes_sentiment.ipynb)

实际计算时, 需要考虑浮点数计算精度的问题. 在概率连乘的时候，因为不同词的概率数量级差别可能很大，容易使计算精度的损失被放大. 解决方法可以是对概率取对数, 把概率的连乘转换为对数的相加.

---

[^1]: Wu et al., 2007, "Top 10 algorithms in data mining"
[^2]: [Speech and Language Processing 第三版, ch6 - Naive Bayes and Sentiment Classification](https://web.stanford.edu/~jurafsky/slp3/6.pdf)
[^3]: 机器学习实战 ch4
