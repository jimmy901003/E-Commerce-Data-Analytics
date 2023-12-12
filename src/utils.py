import json
import pandas as pd
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import operator

# 設定中文字型與風格
plt.rcParams['font.family'] = ['Microsoft YaHei']
sns.set(style="whitegrid")
sns.set_palette("pastel")


# 讀取 JSON 檔案轉換成 DataFrame
def read_json_to_dataframe(file_path):
    data_list = []
    with open(file_path, 'r') as json_file:
        for line in json_file:
            review = json.loads(line)
            data_list.append(review)
    return pd.DataFrame(data_list)


# 提取並修改排名資訊
def extract_and_modify_rank(df):
    df['category'] = df['rank'].str.extract(r'in(.*)\(')[0].str.strip()
    df['rank'] = df['rank'].str.extract(r'([\d,]+)')[0]
    df.rename(columns={'rank': 'product_rank'}, inplace=True)


# 繪製熱力圖
def plot_heatmap(df, cmap="YlGnBu"):
    plt.figure(figsize=(12, 8))
    df_corr = df.corr()
    sns.heatmap(df_corr, cmap=cmap, annot=True, fmt=".2f")
    plt.show()


# 提取 TF-IDF 特徵詞
def extract_tfidf(df, column='reviewText', n=None, custom_stop_words=[]):
    vectorizer = TfidfVectorizer(stop_words='english')
    df_copy = df.copy()
    df_copy.dropna(subset=[column], inplace=True)
    synonym_dict = {
        'fit': ['fits'],
        'shoes': ['shoe', 'shoess'],
        'women': ['womens'],
        'men': ['mens', 'man'],
        'small': ['smaller'],
        'boots': ['boot', 'bootss'],
        'dress': ['dresses'],
        'shirt': ['shirts'],
        'socks': ['sock', 'sockss'],
        'girls': ['girl', 'girlss'],
        'boys': ['boy', 'boyss'],
        'comfortable': ['comfy']
    }

    for key, synonyms in synonym_dict.items():
        for synonym in synonyms:
            df_copy[column] = df_copy[column].str.replace(synonym, key, regex=True, case=False)

    X = vectorizer.fit_transform(df_copy[column])
    feature_names = vectorizer.get_feature_names_out()
    weights = X.sum(axis=0).tolist()[0]
    sorted_weights = sorted(enumerate(weights), key=operator.itemgetter(1), reverse=True)
    idf = vectorizer.idf_

    if n is None:
        n = len(sorted_weights)

    data = []

    for i in range(n):
        index = sorted_weights[i][0]
        word = feature_names[index]
        tfidf = sorted_weights[i][1]
        idf_value = idf[index]
        tf_value = tfidf / idf_value
        if word not in custom_stop_words:
            data.append([word, tfidf, tf_value, idf_value])

    df_result = pd.DataFrame(data, columns=['feature', 'TF-IDF', 'TF', 'IDF'])
    return df_result


# 產生具有 TF-IDF 的文字雲與條形圖
def generate_word_cloud_with_tfidf(data_frame, column='feature', tfidf='TF-IDF', title='word-cloud & TF-IDF BAR', count=20):
    data = dict(zip(data_frame[column], data_frame[tfidf]))
    plt.figure(figsize=(18, 8))
    plt.suptitle(title, fontsize=20)
    
    plt.subplot(121)
    wordcloud = WordCloud(font_path="msyh.ttc", background_color='white', width=500, height=500).generate_from_frequencies(data)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    
    plt.subplot(122)
    sns.barplot(x='TF-IDF', y='feature', data=data_frame.head(count), orient='h')
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()


# 比較評分分佈情況
def compare_rating_distributions(data1, label1, data2, label2, title='Comparison of Relative Frequency of Ratings'):
    plt.rcParams['font.family'] = ['Microsoft YaHei']
    data1_overall = data1['overall'].value_counts(normalize=True)
    data2_overall = data2['overall'].value_counts(normalize=True)

    data1_overall_percent = data1_overall * 100
    data2_overall_percent = data2_overall * 100

    plt.figure(figsize=(8, 6))

    plt.bar(data1_overall.index - 0.15, data1_overall.values, width=0.3, color='skyblue', label=label1)
    plt.bar(data2_overall.index + 0.15, data2_overall.values, width=0.3, color='salmon', label=label2)

    plt.title(title)
    plt.xlabel('Rating')
    plt.ylabel('Relative Frequency (%)')
    plt.xticks(data1_overall.index)
    plt.legend()

    for i, value in enumerate(data1_overall_percent):
        plt.text(data1_overall.index[i] - 0.15, data1_overall.values[i] + 0.01, f'{value:.1f}%', ha='center', va='bottom', color='black', fontsize=8)

    for i, value in enumerate(data2_overall_percent):
        plt.text(data2_overall.index[i] + 0.15, data2_overall.values[i] + 0.01, f'{value:.1f}%', ha='center', va='bottom', color='black', fontsize=8)

    plt.show()

