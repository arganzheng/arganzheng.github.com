---
layout: post
title: 机器学习中的特征工程
tags: [机器学习, 特征工程]
catalog: false
---

* 特征管理
    * 特征元数据注册
        * 特征视图、特征集、特征
        * 数据源信息: 离线数据库、表名称等
        * 存储引擎信息: 引擎类型、机房、IP等
        * 存储格式信息: Key字段、Value字段等
        * 特征更新信息: 更新周期、分区字段、分区方式等
    * 元数据与存储系统打通（如 hive表，Redis 等）
    * 离线特征可视化（类似于 [hue](http://gethue.com/)）
    * 在线特征可视化
    * 特征的共享
        * 共享特征库
        * 引用计数
        * 特征监控和通知
* 特征生产(挖掘)
    * 数据源
        * 内部数据源
            * BI Hive 表
            * 埋点日志上报
            * DB binlog 实时解析和推送
        * 外部数据源
            * Spider
            * 第三方合作
    * 离线特征生产
        * 离线数据源: BI Hive 表
        * 离线计算引擎: Hadoop / Spark
    * 实时特征生产
        * 实时数据源
            * 实时行为日志上报
            * DB binlog 实时解析和推送
        * 实时计算引擎: Strom / Flink
        * 实时特征挖掘
            * 固定时间窗口，时间窗口的起止时间点是固定的，比如某日的销售额。
            * 滑动时间窗口，时间窗口的长度是固定的，但起止时间点一直在向前滚动，比如近2小时销售额。
            * 无限时间窗口，时间窗口的起点是固定的，但终止时间点一直在向前滚动，比如商家历史上销售总额。
* 特征存储
    * Use hashed id instead of string name for efficient
serialization, storage, compute
    * Collocate schema (id to name mapping) with the data
    * 离线特征存储
        * Hive表
        * 命名规范
        * 存储格式
            * 按列存储
            * 打包存储: PB/JSON
        * 元信息
    * 在线特征存储
        * KV: Redis (Anna) / Aerospike / Tair
        * KCV: HBase / Cassandra ([Rocksandra](https://instagram-engineering.com/open-sourcing-a-10x-reduction-in-apache-cassandra-tail-latency-d64f86b43589), [Scylladb](https://www.scylladb.com/)) 
        * 分层存储
        * 压缩
        * 存储格式
            * 按列存储
            * 打包存储: PB/JSON
            * 特征信息
                * name  
                * value 
                * timestamp
                * ttl
                * confidence（置信度）
* 特征上线/更新
    * 分布式特征上线 (Spark)
    * 单模型特征按需上线
    * 特征增量上线方案（差异如何对比？，失效特征如何删除？） 
    * 静态特征 vs. 动态特征 
    * 例行调度
    * 特征上线状态
    * 上线统计 & 监控
    * 原地替换 vs. 双 buffer 切换
    * 分布式流控
* 特征服务
    * RPC
    * 正排 & 倒排 & 高维索引
    * Lamda 架构
    * 强类型的特征服务（根据元数据进行反序列化）
    * 特征缓存
* 特征监控
    * 特征覆盖率
    * 特征异常值
    * 特征时效性
    * 特征监控可视化
    * 监控预警
* 特征分类
    * Low level特征 vs. High level特征
    * 稳定特征与动态特征 （静态特征 vs. 动态特征）
    * 二值特征、连续特征、枚举特征
* 特征抽取和处理
    * 归一化 (Normalization)
        * 将训练集中某一列数值特征（假设是第i列）的值缩放到0和1之间。
        * 归一化方案：
            1. 函数归一化，通过映射函数将特征取值映射到［0，1］区间，例如最大最小值归一化方法，是一种线性的映射。还有通过非线性函数的映射，例如log函数等。
            2. 分维度归一化，可以使用最大最小归一化方法，但是最大最小值选取的是所属类别的最大最小值，即使用的是局部最大最小值，不是全局的最大最小值。
            3. 排序归一化，不管原来的特征取值是什么样的，将特征按大小排序，根据特征所对应的序给予一个新的值。
    * 离散化
        * 等值划分
        * 等量划分
    * 标准化 (Standardization)
        * 将训练集中某一列数值特征（假设是第i列）的值缩放成均值为0，方差为1的状态
        * 具体做法是：将特征值减去平均值，再除以标准差
    * 缺省值处理
        * 单独表示
        * 众数
        * 平均值
    * Embedding
        * word2vec
        * fastext
    * 特征组合
        * 人工特征交叉
        * 模型自动组合
            * GBDT: 最初是由Facebook在2014年提出，并被广泛运用于点击率预估项目上，被证明有效。
            * FM: 可以自动学习两个特征间的关系，可以减少一部分的交叉特征选择工作；但是它无法学习三个及以上的特征间的关系，所以交叉特征选择的工作仍然无法避免。
            * DNN: 可以直接输入原始的特征，减少了交叉特征的选择工作，并且可以支持大量的特征输入。
    * FeatureLibray & FeatureScript
    * 分级抽取：如一次打分请求，用户特征只需抽取一次，item特征则每次都需要抽取
* 特征选择
    * 特征选择能剔除不相关(irrelevant)或冗余(redundant)的特征，从而达到减少特征个数，提高模型精确度，减少运行时间的目的。
    * 三种思路:
        1. Filter: 假设特征子集对模型预估的影响互相独立，选择一个特征子集，分析该子集和数据Label的关系，如果存在某种正相关，则认为该特征子集有效。衡量特征子集和数据Label关系的算法有很多，如Chi-square，Information Gain
            * 卡方
            * 信息增益
            * 交叉熵
            * Fisher
            * FastRegression
        2. Embedded: 将特征选择和模型训练结合起来，如在损失函数中加入L1 Norm ，L2 Norm
            * LOSS + L1
            * 树模型
            * MaxEnt
        3. Wrapper:  选择一个特征子集加入原有特征集合，用模型进行训练，比较子集加入前后的效果，如果效果变好，则认为该特征子集有效，否则认为无效。
            * 遗传法
            * 前向法
            * 后向法
    * 特征有效性分析
    * 特征降维
* 特征分析
    * 单特征 AUC 分析
    * 信息增益率
    * 对特征的有效性进行分析，得到各个特征的特征权重
        1. 与模型相关特征权重。使用所有的特征数据训练出来模型，看在模型中各个特征的权重，由于需要训练出模型，模型相关的权重与此次学习所用的模型比较相关。不同的模型有不同的模型权重衡量方法。例如线性模型中，特征的权重系数等。
        2. 与模型无关特征权重。主要分析特征与label的相关性，这样的分析是与这次学习所使用的模型无关的。与模型无关特征权重分析方法包括
            1. 交叉熵
            2. Information Gain
            3. Odds ratio
            4. 互信息
            5. KL散度等
* 特征降维
    * 特征降维的目标是将高维空间中的数据集映射到低维空间数据，同时尽可能少地丢失信息，或者降维后的数据点尽可能地容易被区分
    * 特征降维方案:
        * PCA : Principal Component Analysis，主成分分析，通过协方差矩阵的特征值分解能够得到数据的主成分
        * LDA : Linear Discriminant Analysis，线性判别分析，与PCA保持数据信息不同，LDA是为了使得降维后的数据点尽可能地容易被区分
        * SVD: Singular Value Decomposition，奇异值分解
        * LLE : Locally Linear Embedding
        * LE : Laplacian Eigenmaps


推荐阅读
------

1. [机器学习中的数据清洗与特征处理综述](https://tech.meituan.com/2015/02/10/machinelearning-data-feature-process.html)
2. [人工智能在线特征系统中的数据存取技术](https://tech.meituan.com/2017/07/06/online-feature-system.html)
3. [人工智能在线特征系统中的生产调度](https://tech.meituan.com/2017/09/22/online-feature-system02.html)
4. [腾讯广告精准投放背后的秘密](https://mp.weixin.qq.com/s?__biz=MjM5MDE0Mjc4MA==&mid=2651016056&idx=2&sn=c88a64c0f943dd0c519b9aec2d387c3a)
5. [Uber 机器学习平台 — 米开朗基罗](https://github.com/xitu/gold-miner/blob/master/TODO/meet-michelangelo-ubers-mechine-learning-plantform.md)
6. [回顾Facebook经典CTR预估模型](https://zhuanlan.zhihu.com/p/57987311)
7. [Xinran He et al. Practical Lessons from Predicting Clicks on Ads at Facebook, 2014](https://quinonero.net/Publications/predicting-clicks-facebook.pdf)
8. [Rocksandra](https://instagram-engineering.com/open-sourcing-a-10x-reduction-in-apache-cassandra-tail-latency-d64f86b43589)
