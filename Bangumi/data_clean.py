import os
import pandas as pd
import jieba
import re

def main():
    raw_path = '/content/Bangumi/raw_data.csv'
    clean_path = '/content/Bangumi/cleaned_data.csv'

    # 增强文件检查
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"原始数据文件 {raw_path} 不存在")
    if os.path.getsize(raw_path) == 0:
        raise ValueError(f"原始数据文件 {raw_path} 为空")

    df = pd.read_csv(raw_path, encoding='utf-8-sig')
    
    # 新增过滤条件
    def clean_text(text):
        text = re.sub(r'@\d{4}-\d{2}-\d{2}', '', str(text))
        text = re.sub(r'$$\d+$$', '', text)
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s。，！？]', '', text)
        words = jieba.lcut(text.strip())
        return ' '.join([w for w in words if len(w) > 1])
    
    # 数据过滤
    df = df[df['raw_score'].between(1, 10)]  # 过滤0分和超范围评分
    df = df.drop_duplicates(subset=['raw_text'])  # 根据文本去重
    df = df.dropna(subset=['raw_text', 'raw_score'])
    
    df['cleaned_text'] = df['raw_text'].apply(clean_text)
    
    # 保存前检查清洗结果
    valid_df = df[df['cleaned_text'].str.len() >= 5]  # 过滤短于5字符的无效文本
    valid_df.to_csv(clean_path, index=False, encoding='utf-8-sig')
    print(f"清洗完成！有效样本数：{len(valid_df)}")

if __name__ == "__main__":
    main()