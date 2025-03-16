import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 配置参数（关键修正）
class Config:
    data_path = "/content/Bangumi/cleaned_data.csv"
    model_save_path = "/content/Bangumi/bangumi_bert"
    batch_size = 16 if torch.cuda.is_available() else 4
    max_len = 64
    epochs = 10 if torch.cuda.is_available() else 3
    lr = 2e-5 if torch.cuda.is_available() else 1e-5

# 数据集类
class BangumiDataset(Dataset):
    def __init__(self, texts, scores, tokenizer):
        self.texts = texts
        self.scores = scores
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=Config.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.scores[idx], dtype=torch.float)  # 回归任务必须用float
        }

def main():
    # 设备检测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前运行设备：{device}")
    
    # 数据存在性检查
    if not os.path.exists(Config.data_path):
        raise FileNotFoundError(f"清洗数据文件 {Config.data_path} 不存在")

    # 加载数据
    df = pd.read_csv(Config.data_path, encoding='utf-8-sig')
    texts = df['cleaned_text'].tolist()
    scores = df['raw_score'].astype(float).tolist()
    
    # 划分数据集
    train_texts, val_texts, train_scores, val_scores = train_test_split(
        texts, scores, test_size=0.2, random_state=42
    )
    
    # 初始化模型（关键修正点）
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-chinese",
        num_labels=1,
        problem_type="regression"  # 必须明确指定回归任务！
    ).to(device)
    
    # 数据加载器
    train_dataset = BangumiDataset(train_texts, train_scores, tokenizer)
    val_dataset = BangumiDataset(val_texts, val_scores, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size)
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=Config.lr)
    
    # 训练循环
    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{Config.epochs} | 训练损失：{total_loss/len(train_loader):.4f}")
        
        # 验证集评估（关键：输出原始分数）
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(device)
                outputs = model(**inputs)
                preds.extend(outputs.logits.squeeze().cpu().numpy())
                truths.extend(labels.cpu().numpy())
        
        mae = mean_absolute_error(truths, preds)
        mse = mean_squared_error(truths, preds)
        print(f"验证集 MAE：{mae:.3f}（平均误差 ±{mae:.1f}分）")
        print(f"验证集 MSE：{mse:.3f}\n")
    
    # 保存模型
    os.makedirs(Config.model_save_path, exist_ok=True)
    model.save_pretrained(Config.model_save_path)
    tokenizer.save_pretrained(Config.model_save_path)
    print(f"模型已保存至：{Config.model_save_path}")

if __name__ == "__main__":
    main()