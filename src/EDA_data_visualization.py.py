import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_processing import merged_df, df_reviews, df_meta


# 計算並繪製缺失值百分比的柱狀圖
def plot_missing_values(df, title='Missing Values Percentage'):
    missing_data = df.isnull().mean() * 100
    missing_data = missing_data.sort_values(ascending=False)

    plt.figure(figsize=(12, 10))
    plot = sns.barplot(x=missing_data.values, y=missing_data.index, palette='Set2')

    plt.title(title, fontsize=16)
    plt.xlabel('Percentage of Missing Values')
    plt.ylabel('Columns')
    plt.xlim(0, 100)
    plt.grid(axis='x')

    plot.tick_params(axis='y', labelsize=16)

    for i, v in enumerate(missing_data.values):
        plot.text(v + 0.3, i, f'{v:.1f}%', color='black', va='center')

    plt.tight_layout()
    plt.show()

# 繪製評論和元數據 DataFrame 的缺失值
plot_missing_values(df_reviews, title='Missing Values Percentage in Reviews')
plot_missing_values(df_meta, title='Missing Values Percentage in Meta Data')

# 繪製整體評分值的計數
plt.figure(figsize=(10, 6))

ax = sns.countplot(data=merged_df, x='overall', palette='Set2')

plt.title('Overall Ratings Count', fontsize=16)
plt.xlabel('Overall Rating')
plt.ylabel('Count')

total_count = len(merged_df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2, height + 5, f'{height/total_count:.2%}', ha='center')

plt.show()

# 繪製每月整體評分的趨勢
plt.figure(figsize=(12, 6))

monthly_counts = merged_df['overall'].resample('M').sum()
sns.lineplot(x=monthly_counts.index, y=monthly_counts.values, marker='o', markersize=8, color='b', alpha=0.7)

plt.title('Monthly Overall Ratings Trend')
plt.xlabel('Month')
plt.ylabel('Total Overall Ratings')

plt.show()

# 繪製已驗證評論的百分比
plt.figure(figsize=(8, 6))

ax = sns.countplot(data=df_reviews, x='verified', palette='Set2')

plt.title('Percentage of Verified Reviews', fontsize=16)
plt.xlabel('Verified', fontsize=12)
plt.ylabel('Percentage', fontsize=12)

total = len(df_reviews['verified'])

for p in ax.patches:
    height = p.get_height()
    percentage = (height / total) * 100
    ax.text(p.get_x() + p.get_width() / 2., height + 0.1, f'{percentage:.2f}%', ha='center', fontsize=10)

plt.show()
