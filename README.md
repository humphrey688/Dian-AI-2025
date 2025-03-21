# Dian-AI-2025
## ⼀、随机森林的理解与实现说明
1. **随机森林模型构建**  
- **采用** CART 决策树（基于 Gini 指数分裂）
- **集成方法**：Bootstrap 采样 + 随机特征子集（√n_features）
- **预测方式**：多数投票机制

2. **模型训练与评估**  
- **数据集**：Iris（150样本，4特征，3分类）
- **划分**：80%训练集（120条）/20%测试集（30条）
- **评估指标**：准确率

3. **特征重要性分析**  
- **方法**：基于置换重要性（Permutation Importance）
- **可视化**：生成 `Figure_1.png`

## 二、Bangumi评论分数预测器的训练说明
1. **模型说明** 
- **任务**: 基于BERT的Bangumi评论评分预测（10分制）
- **训练框架**: PyTorch
- **模型架构**: bert-base-chinese

2. **代码结构** 
- **数据爬取**: Bangumi/crawler.py
- **数据清洗**: Bangumi/data_clean.py
- **模型训练**: Bangumi/train.py

3. **模型下载**
- **Hugging Face仓库名称**: humphrey688/bangumi-bert-rating
- **仓库链接**: https://huggingface.co/humphrey688/bangumi-bert-rating

## 三、注意⼒机制及其变体的理解与实现说明
1. **标准多头注意力（MHA）** 
- **原理**: 将输入分别通过线性变换得到多组查询（Query）、键（Key）、值（Value）向量。对每组 Query 与 Key 计算点积，缩放后经 SoftMax 得到注意力权重，再与对应 Value 加权求和，最后拼接多头结果并线性变换输出。这样能让模型从不同子空间捕捉信息，公式为：
$$Attention(Q,K,V)=\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$
其中， ${d_k}$  是每个头的维度。
2. **GQA与MQA** 
- **GQA原理**: MQA 中所有头共享相同的键和值。查询向量经线性变换后分头，键值向量经线性变换后拆分并重复多头。后续计算与 MHA 类似，计算点积、缩放、掩码（若有）、SoftMax 得到注意力权重。因其共享键值，减少了内存占用和计算量。
- **MQA原理**: GQA 把多头注意力的头分组，每组共享键和值。查询向量线性变换后按组拆分，对每组分别计算查询与键的点积，缩放、掩码（若有）后经 SoftMax 得到组内注意力权重，最后拼接所有组的加权结果。这种方式减少了键值对的存储量。
