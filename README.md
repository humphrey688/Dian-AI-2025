# Dian-AI-2025
## ⼀、随机森林的理解与实现代码说明
1. **随机森林模型构建**  
   - **采用** CART 决策树（基于 Gini 指数分裂）
   - **集成方法**：Bootstrap 采样 + 随机特征子集（√n_features）
   - **预测方式**：多数投票机制

2. **模型训练与评估**  
   - **数据集**：Iris（150样本，4特征，3分类）
   - **划分**：80%训练集（120条）/20%测试集（30条）
   - **评估指标**：准确率（最终测试集准确率 100%）

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
