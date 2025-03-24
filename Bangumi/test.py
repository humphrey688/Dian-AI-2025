import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification  

model_name = "humphrey688/bangumi-bert-rating"

# 加载分词器和模型
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)  # 按实际任务类型修改模型类
except Exception as e:
    print(f"加载模型出错：{e}")
    exit()

# 构造测试输入
text = "这部番剧非常精彩"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# 运行模型
with torch.no_grad():
    outputs = model(**inputs)
    # 将 Tensor 转换为 Python 数值
    rating = outputs.logits.item()
    print(f"该番剧的预测评分为: {rating}")
