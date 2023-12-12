import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from data_processing import (merged_df, product)
from utils import (extract_tfidf,
                  generate_word_cloud_with_tfidf,  
                  plot_heatmap, read_json_to_dataframe, 
                  extract_and_modify_rank,
                  compare_rating_distributions
                  )

plt.rcParams['font.family'] = ['Microsoft YaHei'] 

# 設定 pandas 顯示的最大列數
pd.set_option("display.max_columns", 10000)
pd.set_option("display.max_rows", 10000)


# 根據產品評分篩選
high_overall_p = product[product['product_overall_mean'] >= 4.5]
low_overall_p = product[product['product_overall_mean'] < 3]

# 根據交叉購買量篩選
high_also_buy_p = product[product['also_buy_count'] >= 40]
low_also_buy_p = product[product['also_buy_count'] <= 30]

# 篩選2018年交叉購買量
high_also_buy_2018 = merged_df[(merged_df['also_buy_count'] >= 40) & (merged_df['year'] == 2018)]
high_also_buy_2018_ids = set(high_also_buy_2018['asin'].unique().tolist())
high_also_buy_2018_p = product[product['asin'].isin(high_also_buy_2018_ids)]

low_also_buy_2018 = merged_df[(merged_df['also_buy_count'].between(1, 5)) & (merged_df['year'] == 2018)]
low_also_buy_2018_ids = set(low_also_buy_2018['asin'].unique().tolist())
low_also_buy_2018_p = product[product['asin'].isin(low_also_buy_2018_ids)]

# 分類2018年交叉購買量高低和產品評分
g_low_also_buy_2018_p = low_also_buy_2018_p[low_also_buy_2018_p['product_overall_mean'] >= 4.5]
b_low_also_buy_2018_p = low_also_buy_2018_p[low_also_buy_2018_p['product_overall_mean'] < 3]

g_high_also_buy_2018_p = high_also_buy_2018_p[high_also_buy_2018_p['product_overall_mean'] >= 4.5]
b_high_also_buy_2018_p = high_also_buy_2018_p[high_also_buy_2018_p['product_overall_mean'] < 3]


'also_buy_count與評分關聯性'
plt.figure(figsize=(14, 8))

years_to_plot = [2014, 2015, 2016, 2017, 2018]  

# 繪製每年的also_buy_count與overall之間的關聯性圖表
for year in years_to_plot:
    data_for_year = merged_df[merged_df['year'] == year]
    sns.lineplot(x='overall', y='also_buy_count', data=data_for_year, label=f'Year {year}')


plt.title('Comparison of also_buy_count by Year')
plt.xlabel('overall')
plt.ylabel('also_buy_count')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()

#統計驗證 
model = smf.ols('also_buy_count ~ overall', data=merged_df).fit()
print(model.summary())

'高評分產品高低交叉購買率產品數量統計'
g_high_also_buy_2018_p = high_also_buy_2018_p[high_also_buy_2018_p['product_overall_mean']>=4.5]
g_low_also_buy_2018_p = low_also_buy_2018_p[low_also_buy_2018_p['product_overall_mean']>=4.5]

product_counts = [len(g_high_also_buy_2018_p), len(g_low_also_buy_2018_p)]
product_labels = ['高交叉購買率', '低交叉購買率']  

plt.figure(figsize=(8, 6))

sns.barplot(x=product_labels, y=product_counts, palette=['skyblue', 'salmon'])

plt.title('高評分產品高低交叉購買率產品數量統計')
plt.xlabel('Products')
plt.ylabel('Counts')
plt.show()

'高低交叉購買率產品評論數量統計'
high_product_log = np.log(high_also_buy_2018_p['product_rating_count'] + 1)  
low_product_log = np.log(low_also_buy_2018_p['product_rating_count'] + 1)

plt.figure(figsize=(8, 6))

plt.boxplot([high_product_log, low_product_log], labels=['高交叉購買率', '低交叉購買率'])
plt.title('高低交叉購買率產品評論數量統計')
plt.ylabel('Log(Product Rating Count)')
plt.show()

'高低平留言顧客評分分布'
high_comment = merged_df[(merged_df['comment_count']>=3)]
low_comment = merged_df[(merged_df['comment_count']<=2)]
compare_rating_distributions(high_comment, '高頻留言顧客評分', low_comment, '低頻留言顧客評分' )

model = smf.ols('comment_count ~ overall', data=merged_df).fit()
print(model.summary())


'tfidf過濾字'
good_custom_stop_words = [
    'case', 'good', 'great', 'nice', 'love', 'perfect', 'loved', 'like', 
    'just', 'loves', 'looks', 'look', 'did', 'really', 'super', 'got', 
    'bought', 'buy', 'looking'
]

bad_custom_stop_words = [
    'don', 'like', 'looks', 'look', 'just', 'didn', 'good', 'did', 
    'really', 'does', 'bought', 'gave', 'buy', 'looking'
]

title_custom_stop_words = [
    'print', 'set', 'xl'
]

gb_custom_stop_words = [
    'case', 'good', 'great', 'nice', 'love', 'perfect', 'loved', 'like', 
    'just', 'loves', 'looks', 'look', 'did', 'really', 'super', 'got', 
    'bought', 'buy', 'looking', 'don', 'like', 'didn', 'did', 'does', 'gave'
]

'好評留言TFIDF'
good_reviews_df = merged_df[merged_df['overall'] >= 4.5]
good_reviews_tfidf  = extract_tfidf(good_reviews_df, custom_stop_words=good_custom_stop_words)
generate_word_cloud_with_tfidf(good_reviews_tfidf, title='Positive Reviews')


'cute的評分分布'
cute = merged_df[merged_df['reviewText'].str.contains('cute', case=False)]
compare_rating_distributions(cute,'含有cute的評論評分', merged_df,'整體服飾市場評分')


'高分產品標題TFIDF'
high_overall_p_tfidf = extract_tfidf(high_overall_p, column='title', custom_stop_words=title_custom_stop_words)
generate_word_cloud_with_tfidf(high_overall_p_tfidf, title='high overall product tfidf')


'含 cotton 與 leather 的產品數量統計'
leather_p = product[product['title'].str.contains('leather', case=False)]
cotton_p = product[product['title'].str.contains('cotton', case=False)]

count_above_4_5_cotton_p = (cotton_p['product_overall_mean'] >= 4.5).sum()
count_below_3_cotton_p = (cotton_p['product_overall_mean'] < 3).sum()

count_above_4_5_leather_p = (leather_p['product_overall_mean'] >= 4.5).sum()
count_below_3_leather_p = (leather_p['product_overall_mean'] < 3).sum()

counts_cotton = [count_above_4_5_cotton_p, count_below_3_cotton_p]
counts_leather = [count_above_4_5_leather_p , count_below_3_leather_p]

labels = ['>=4.5', '< 3']
plt.figure(figsize=(8, 6))
sns.barplot(x=labels, y=counts_cotton, color='skyblue', label='cotton')
sns.barplot(x=labels, y=counts_leather, color='salmon', bottom=counts_leather, label='leather')

plt.xlabel('評分範圍')
plt.ylabel('數量')
plt.title('含 cotton 與 leather 的產品數量統計')
plt.legend()

plt.show()


'高分cotton產品的好評TFIDF'
cotton = merged_df[merged_df['title'].str.contains('cotton', case=False)]
g_cotton = cotton[(cotton['product_overall_mean']>=4.5) & (cotton['overall']>=4)]

g_cotton_tfidf = extract_tfidf(g_cotton, custom_stop_words=good_custom_stop_words)
generate_word_cloud_with_tfidf(g_cotton_tfidf, title='Positive Reviews of High-Rated Cotton Products Purchase')


'cotton材質中含有cute評分分布'
cotton_cute = cotton[(cotton['reviewText'].str.contains('cute', case=False)) ]
compare_rating_distributions(cotton_cute,'cotton材質中含有cute的評分', merged_df,'整體服飾市場評分')


'高分leather產品的好評TFIDF'
leather = merged_df[merged_df['title'].str.contains('leather', case=False)]
g_leather = leather[(leather['product_overall_mean']>=4.5) & (leather['overall']>=4)]

g_leather_tfidf = extract_tfidf(g_leather, custom_stop_words=good_custom_stop_words)
generate_word_cloud_with_tfidf(g_leather_tfidf, title='Positive Reviews of High-Rated leather Products Purchase')

'2018年 high_also_buy產品標題TFIDF'
high_also_buy_2018=merged_df[(merged_df['also_buy_count'] >=40) & (merged_df['year'] ==2018) ]
high_also_buy_2018_ids = set(high_also_buy_2018['asin'].unique().tolist())
high_also_buy_2018_p = product[product['asin'].isin(high_also_buy_2018_ids)]

high_also_buy_2018_p_tfidf = extract_tfidf(high_also_buy_2018_p, column='title', custom_stop_words=title_custom_stop_words)
generate_word_cloud_with_tfidf(high_also_buy_2018_p_tfidf, title='high also buy product title')

'負評留言TFIDF'
bad_reviews_df = merged_df[merged_df['overall'] < 3]
bad_reviews_tfidf  = extract_tfidf(bad_reviews_df, custom_stop_words=bad_custom_stop_words)
generate_word_cloud_with_tfidf(bad_reviews_tfidf, title='Negative Reviews')

'計算三種負評比例'
small = bad_reviews_df[bad_reviews_df['reviewText'].str.contains('small', case=False) ]
big = bad_reviews_df[(bad_reviews_df['reviewText'].str.contains('large', case=False)) | (bad_reviews_df['reviewText'].str.contains('big', case=False))]
quality= bad_reviews_df[(bad_reviews_df['reviewText'].str.contains('quality', case=False)) | (bad_reviews_df['reviewText'].str.contains('material', case=False))]


product_counts = [len(small) / len(bad_reviews_df), len(big) / len(bad_reviews_df), len(quality) / len(bad_reviews_df)]
product_labels = ['尺碼過小', '尺碼過大', '劣質品質、材質']

product_percentages = [count * 100 for count in product_counts]

plt.figure(figsize=(8, 8))

ax = sns.barplot(x=product_labels, y=product_counts, palette=['skyblue', 'salmon', '#72fac4'])

for i, percentage in enumerate(product_percentages):
    plt.text(i, product_counts[i] + 0.005, f'{percentage:.1f}%', ha='center', va='bottom', color='black', fontsize=10)


plt.title('低分原因占比', fontsize=16)
plt.xlabel('Problem')
plt.ylabel('Percentage')

plt.show()

'低分產品標題TFIDF'
low_overall_p=product[product['product_overall_mean'] <=2]

low_overall_p_tfidf  = extract_tfidf(low_overall_p, column='title', custom_stop_words=title_custom_stop_words)
generate_word_cloud_with_tfidf(low_overall_p_tfidf, title='low overall product title')

'購買chiffon評論TFIDF'
chiffon = merged_df[merged_df['title'].str.contains('chiffon', case=False)]
chiffon_tfidf  = extract_tfidf(chiffon, column='title', custom_stop_words=gb_custom_stop_words)
generate_word_cloud_with_tfidf(chiffon_tfidf, title='Reviews for purchasing chiffon')

'chiffon材質中dress與shirt數量'
chiffon_p = product[product['title'].str.contains('chiffon', case=False)]
chiffon_dress_p = chiffon_p[chiffon_p['title'].str.contains('dress', case=False)]
chiffon_shirt_p = chiffon_p[chiffon_p['title'].str.contains('shirt', case=False)]

'計算含有 "dress" 的評分範圍數量'
count_above_4_5_chiffon_dress = (chiffon_dress_p['product_overall_mean'] >= 4.5).sum()
count_below_3_chiffon_dress = (chiffon_dress_p['product_overall_mean'] < 3).sum()

'計算含有 "shirt" 的評分範圍數量'
count_above_4_5_chiffon_shirt = (chiffon_shirt_p['product_overall_mean'] >= 4.5).sum()
count_below_3_chiffon_short = (chiffon_shirt_p['product_overall_mean'] < 3).sum()


'無 "dress" "shirt" 評分範圍數量'
count_above_4_5 = (chiffon_p['product_overall_mean'] >= 4.5).sum() - count_above_4_5_chiffon_dress - count_above_4_5_chiffon_shirt
count_below_3 = (chiffon_p['product_overall_mean'] < 3).sum() - count_below_3_chiffon_dress - count_above_4_5_chiffon_shirt


counts_dress = [count_above_4_5_chiffon_dress, count_below_3_chiffon_dress]
counts_shirt = [count_above_4_5_chiffon_shirt, count_below_3_chiffon_short]
counts_other = [count_above_4_5, count_below_3]
labels = ['>=4.5', '< 3']

plt.figure(figsize=(8, 6))

sns.barplot(x=labels, y=counts_other, color='skyblue', label='其他')
sns.barplot(x=labels, y=counts_dress, color='salmon', bottom=counts_other, label='dress')

bottom_combined = [counts_other[i] + counts_dress[i] for i in range(len(counts_other))]
sns.barplot(x=labels, y=counts_shirt, color='#72fac4', bottom=bottom_combined, label='shirt')

plt.xlabel('評分範圍')
plt.ylabel('數量')
plt.title('chiffon中含 dress 與含 shirt 的評分範圍數量統計')
plt.legend()

plt.show()

'chiffin評論數量分布'
plt.figure(figsize=(8, 6))

'dress不含chiffon評分分布'
dress = merged_df[merged_df['title'].str.contains('dress', case=False)]
no_chiffon_dress = merged_df[(merged_df['title'].str.contains('dress', case=False)) & (~merged_df['title'].str.contains('chiffon', case=False))]
compare_rating_distributions(dress, 'dress類服飾評分', no_chiffon_dress, '不含chiffon材質的dress服飾評分')

'dress不含chiffon文字雲'
b_dress = dress[dress['product_overall_mean']<3.0]

dress_tfidf = extract_tfidf(dress)
generate_word_cloud_with_tfidf(dress_tfidf)

no_chiffon_dress_tfidf = extract_tfidf(no_chiffon_dress)
generate_word_cloud_with_tfidf(no_chiffon_dress_tfidf)

'dress 負評'
b_dress = dress[dress['overall']<3.0]

b_dress_tfidf = extract_tfidf(no_chiffon_dress, custom_stop_words=bad_custom_stop_words)
generate_word_cloud_with_tfidf(b_dress_tfidf, title='Negative Reviews for purchasing dress')

'高交叉率產品品牌回頭客率'
brand_customer_purchase_count = high_also_buy_2018.groupby(['brand', 'reviewerID']).size().reset_index(name='purchase_count')

#篩選出重複購買超過一次的顧客
brand_repeated_customers = brand_customer_purchase_count[brand_customer_purchase_count['purchase_count'] >= 2].groupby('brand')['reviewerID'].nunique()

# 計算重複購買顧客佔比
high_also_buy_2018 = high_also_buy_2018.copy()
high_also_buy_2018 = high_also_buy_2018[high_also_buy_2018['brand_rating_count']>=10]
brand_total_customers = high_also_buy_2018.groupby('brand')['reviewerID'].nunique()
brand_repeated_percentage = (brand_repeated_customers / brand_total_customers).fillna(0) * 100

brand_repeated_percentage_sorted = pd.DataFrame(brand_repeated_percentage.sort_values(ascending=False))

greater_than_5 = brand_repeated_percentage_sorted[brand_repeated_percentage_sorted['reviewerID'] >= 5]
less_than_5 = brand_repeated_percentage_sorted[brand_repeated_percentage_sorted['reviewerID'] < 5]

# 計算大於和小於 5 的數量
count_greater_than_5 = len(greater_than_5)
count_less_than_5 = len(less_than_5)

percentage_greater_than_5 = (count_greater_than_5 / len(brand_repeated_percentage_sorted)) * 100
percentage_less_than_5 = (count_less_than_5 / len(brand_repeated_percentage_sorted)) * 100


plt.figure(figsize=(8, 6))

sns.barplot(x=['回頭客率大於等於5的品牌', '回頭客率小於5的品牌'], y=[percentage_greater_than_5, percentage_less_than_5], palette=['skyblue', 'salmon'])

for index, value in enumerate([percentage_greater_than_5, percentage_less_than_5]):
    plt.text(index, value + 1, f'{value:.2f}%', ha='center')

plt.title('高交叉率產品品牌回頭客率')
plt.xlabel('分類')
plt.ylabel('產品數量百分比')
plt.ylim(0, 100)  
plt.show()


