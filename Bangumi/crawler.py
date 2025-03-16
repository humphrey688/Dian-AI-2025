import os
import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
import random

def main():
    os.makedirs('/content/Bangumi', exist_ok=True)
    raw_path = '/content/Bangumi/raw_data.csv'
    
    # 加载历史数据
    if os.path.exists(raw_path) and os.path.getsize(raw_path) > 0:
        existing_df = pd.read_csv(raw_path)
        # 兼容处理旧数据没有subject_id的情况
        if 'subject_id' in existing_df.columns:
            existing_ids = set(existing_df['subject_id'].unique())
        else:
            existing_ids = set()
            print("检测到旧版数据格式，自动重置ID记录")
    else:
        existing_ids = set()
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://bangumi.tv/'
    }
    
    proxies = {
        'http': 'http://127.0.0.1:10809',
        'https': 'http://127.0.0.1:10809'
    } if os.environ.get('COLAB_RELEASE_TAG') is None else None
    
    new_comments = []
    
    # 仅抓取新条目（1-100）
    for subject_id in range(1, 101):
        if subject_id in existing_ids:
            continue
            
        try:
            url = f"https://bangumi.tv/subject/{subject_id}/comments"
            resp = requests.get(url, headers=headers, proxies=proxies, timeout=15)
            resp.encoding = 'utf-8'
            
            if "安全验证" in resp.text:
                print("触发反爬机制，终止爬取")
                break
            
            soup = BeautifulSoup(resp.text, 'html.parser')
            items = soup.select('div.item')
            
            if not items:
                print(f"条目 {subject_id} 无评论")
                continue
                
            for item in items:
                text_elem = item.select_one('div.text')
                rating_elem = item.select_one('span.starlight')
                if text_elem and rating_elem:
                    text = text_elem.get_text(strip=True)
                    rating = rating_elem['class'][-1].replace('stars', '')
                    new_comments.append({
                        'subject_id': subject_id,
                        'raw_text': text,
                        'raw_score': rating
                    })
                    print(f"已抓取条目 {subject_id} 评论：{text[:20]}... 评分：{rating}")
            
            # 随机延时
            delay = 3 + random.random()*3 if proxies else 1 + random.random()*2
            time.sleep(delay)
            
        except Exception as e:
            print(f"条目 {subject_id} 错误：{str(e)[:50]}")
            time.sleep(10)
    
    # 追加保存新数据
    if new_comments:
        new_df = pd.DataFrame(new_comments)
        if os.path.exists(raw_path):
            old_df = pd.read_csv(raw_path)
            # 合并时自动处理列差异
            combined_df = pd.concat([old_df, new_df], ignore_index=True, sort=False)
        else:
            combined_df = new_df
            
        combined_df.to_csv(raw_path, index=False, encoding='utf-8-sig')
        print(f"新增 {len(new_df)} 条数据，总数据量：{len(combined_df)}")
    else:
        print("未爬取到新数据")

if __name__ == "__main__":
    main()