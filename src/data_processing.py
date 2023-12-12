import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import read_json_to_dataframe, extract_and_modify_rank


plt.rcParams['font.family'] = ['Microsoft YaHei']
# pd.set_option("display.max_columns", 10000)
# pd.set_option("display.max_rows", 10000)

# 讀取 JSON 檔案成為 DataFrame
reviews_file_path = r"E:\project\AmazonAnalyticsProject\data\AMAZON_FASHION.json"
meta_file_path = r"E:\project\AmazonAnalyticsProject\data\meta_AMAZON_FASHION.json"

df_reviews = read_json_to_dataframe(reviews_file_path)
df_meta = read_json_to_dataframe(meta_file_path)
df_meta2 = df_meta.copy()

# meta前處理
df_meta2['feature'] = df_meta2['feature'].fillna('NOFEATURE')
df_meta2['feature'] = df_meta2['feature'].apply(lambda x: x[0])

# 新增產品關聯資訊欄位
df_meta2['also_buy_count'] = df_meta2['also_buy'].apply(lambda x: len(x) if isinstance(x, list) else 0)
df_meta2['also_view_count'] = df_meta2['also_view'].apply(lambda x: len(x) if isinstance(x, list) else 0)

df_meta2.drop(['fit', 'tech1', 'imageURL', 'date', 'similar_item', 'details'], axis=1, inplace=True)

# 處理排名欄位
extract_and_modify_rank(df_meta2)
df_meta2 = df_meta2.dropna(subset=['product_rank'])
df_meta2['product_rank'] = df_meta2['product_rank'].str.replace(',', '').astype(int)

# 合併評論資料
merged_df = df_reviews.merge(df_meta2, on='asin', how='inner')

# 處理日期欄位
merged_df['reviewTime'] = pd.to_datetime(merged_df['reviewTime'], format='%m %d, %Y')
merged_df.set_index('reviewTime', inplace=True)
merged_df.sort_index(inplace=True)

# 處理空值欄位
merged_df = merged_df.dropna(subset=['reviewText'])
merged_df['vote'].fillna(0, inplace=True)
merged_df['vote'] = pd.to_numeric(merged_df['vote'])
merged_df['brand'] = merged_df['brand'].fillna('No Brand')
df_meta2['title'] = df_meta2['title'].fillna('').apply(str)
merged_df['title'] = merged_df['title'].fillna('').apply(str)

# 新增產品資訊欄位
merged_df['product_overall_mean'] = merged_df.groupby('asin')['overall'].transform('mean')
merged_df['product_rating_count'] = merged_df.groupby('asin')['overall'].transform('count')
merged_df['brand_rating_count'] = merged_df.groupby('brand')['overall'].transform('count')

# 處理價格欄位
prices = pd.to_numeric(merged_df['price'].str.extractall(r'\$([\d.]+)')[0], errors='coerce')
prices_df = prices.reset_index(level='match')
prices_df.columns = ['match', 'price']
avg_prices = prices_df.groupby(prices_df.index)['price'].mean()
merged_df['price'] = merged_df.index.map(avg_prices)

# 篩選年分資料
years_to_remove = [2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013]
merged_df = merged_df[~merged_df.index.year.isin(years_to_remove)]

merged_df['year'] = merged_df.index.year

# 新增顧客評論資訊欄位
merged_df['original_index'] = merged_df.index
customer_comment_count = merged_df.groupby('reviewerID').size().reset_index(name='comment_count')
average_rating_by_reviewer = merged_df.groupby('reviewerID')['overall'].mean().reset_index(name='average_rating')
merged_df = merged_df.merge(average_rating_by_reviewer, on='reviewerID', how='left')
merged_df = merged_df.merge(customer_comment_count, on='reviewerID', how='left')
merged_df.set_index('original_index', inplace=True)

# 完整產品data
product = (
    merged_df.groupby(['asin'])
    .agg({'product_rating_count': 'mean', 'product_overall_mean': 'mean'})
    .reset_index()
    .merge(df_meta2, on='asin', how='inner')
)
