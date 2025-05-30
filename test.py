import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter, defaultdict
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.cluster import KMeans
import os

# ==== 1. 读取CSV文件 ====
df = pd.read_csv('comments.csv')
texts = df['comment'].astype(str).fillna('').tolist()

# ==== 2. 加载情感词典 ====
def load_sentiment_dict(path):
    sentiment_dict = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                word, score = parts
                try:
                    sentiment_dict[word] = float(score)
                except ValueError:
                    continue
    return sentiment_dict

sentiment_dict = load_sentiment_dict('korean_sentiment_lexicon.txt')

# ==== 3. 最大匹配分词器 ====
class MaxMatchTokenizer:
    def __init__(self, vocab):
        self.vocab = set(vocab)
        self.maxlen = max(len(w) for w in self.vocab) if self.vocab else 0

    def tokenize(self, text):
        tokens = []
        idx = 0
        length = len(text)
        while idx < length:
            matched = False
            for l in range(self.maxlen, 0, -1):
                if idx + l > length:
                    continue
                piece = text[idx:idx+l]
                if piece in self.vocab:
                    tokens.append(piece)
                    idx += l
                    matched = True
                    break
            if not matched:
                idx += 1
        return tokens

tokenizer = MaxMatchTokenizer(sentiment_dict.keys())

# ==== 4. 分词并计算情感分数 ====
def sentiment_score(tokens):
    return sum(sentiment_dict.get(token, 0) for token in tokens)

def sentiment_hit(tokens):
    return sum(1 for token in tokens if token in sentiment_dict)

tokenized_texts = [tokenizer.tokenize(text) for text in texts]
df['tokens'] = tokenized_texts
df['sentiment'] = [sentiment_score(tokens) for tokens in tokenized_texts]
df['sentiment_hit'] = [sentiment_hit(tokens) for tokens in tokenized_texts]

# ==== 5. 停用词处理 ====
basic_stopwords = set([
    '이', '그', '저', '것', '수', '등', '들', '의', '가', '을', '를', '에', '에서', '에게', '한', '하다',
    '및', '그리고', '더', '또한', '또', '하지만', '그러나', '때문에', '하지만', '로', '으로', '와', '과',
    '이다', '있는', '하여', '하고', '된다', '까지', '부터', '나', '너', '우리', '너희', '여기', '저기',
    '그것', '저것', '이것', '이런', '그런', '저런', '그래서', '그러므로', '즉', '예를', '예를들면', '때', '시',
    '년', '월', '일', '시간', '분', '초', '있다', '없다', '이다', '된다', '삭제된', '댓글입니다', '삭제일시'
])
basic_stopwords.update([str(i) for i in range(0, 10000)])

# ==== 6. 词频统计 ====
all_words = [w for tokens in tokenized_texts for w in tokens if w not in basic_stopwords and not w.isdigit() and len(w) > 1]
word_freq = Counter(all_words)
top20_words = word_freq.most_common(20)

# ==== 7. 自动查找字体路径（韩文/中日韩字体）====
def get_font_path():
    possible_fonts = [
        'C:/Windows/Fonts/malgun.ttf',
        'C:/Windows/Fonts/malgunbd.ttf',
        'C:/Windows/Fonts/gulim.ttc',
        'C:/Windows/Fonts/batang.ttc',
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        '/System/Library/Fonts/AppleSDGothicNeo.ttc'
    ]
    for font in possible_fonts:
        if os.path.exists(font):
            return font
    raise Exception("找不到韩文字体文件，请手动指定 font_path")

font_path = get_font_path()

# ==== 8. 词云绘制 ====
wc = WordCloud(font_path=font_path, width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
plt.figure(figsize=(12,6))
plt.imshow(wc, interpolation="bilinear")
plt.axis('off')
plt.title('WordCloud')
plt.show()

# ==== 9. 情感分数直方图 ====
plt.figure(figsize=(8,6))
plt.hist(df['sentiment'], bins=20, color='skyblue', edgecolor='black')
plt.title("Sentiment Score Distribution")
plt.xlabel("Sentiment Score")
plt.ylabel("Number of Comments")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ==== 10. 共现网络分析 ====
def build_cooccurrence_network(tokenized_texts, min_edge=1):
    cooccur = defaultdict(int)
    for tokens in tokenized_texts:
        unique_tokens = set([w for w in tokens if w not in basic_stopwords and not w.isdigit() and len(w) > 1])
        for w1 in unique_tokens:
            for w2 in unique_tokens:
                if w1 < w2:
                    pair = (w1, w2)
                    cooccur[pair] += 1
    G = nx.Graph()
    for (w1, w2), freq in cooccur.items():
        if freq >= min_edge:
            G.add_edge(w1, w2, weight=freq)
    return G

G = build_cooccurrence_network(tokenized_texts, min_edge=1)
cooccur_node_count = G.number_of_nodes()
cooccur_edge_count = G.number_of_edges()

if cooccur_node_count == 0 or cooccur_edge_count == 0:
    print("共现网络中没有足够的节点或边，请检查分词、停用词设置，或降低min_edge阈值。")
else:
    plt.figure(figsize=(10,10))
    pos = nx.spring_layout(G, k=0.8 if G.number_of_nodes() < 50 else 0.5)
    nx.draw(G, pos, node_size=80 if G.number_of_nodes() < 30 else 40,
            alpha=0.7, edge_color='gray', with_labels=False)
    top_nodes = dict(sorted(G.degree, key=lambda x: x[1], reverse=True)[:30])
    nx.draw_networkx_labels(G, pos, labels={n: n for n in top_nodes},
                           font_size=12, font_family='Malgun Gothic', font_color='black')
    plt.title('Co-occurrence Network')
    plt.axis('off')
    plt.show()

# ==== 11. 主题建模（LDA）====
texts_for_vector = [' '.join([w for w in tokens if w not in basic_stopwords and not w.isdigit() and len(w)>1]) for tokens in tokenized_texts]
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=list(basic_stopwords))
tf = tf_vectorizer.fit_transform(texts_for_vector)
num_topics = 5
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(tf)
feature_names = tf_vectorizer.get_feature_names_out()

def extract_top_words(model, feature_names, n_top_words=10):
    topic_words = []
    for topic_idx, topic in enumerate(model.components_):
        words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topic_words.append((topic_idx, words))
    return topic_words

lda_topics = extract_top_words(lda, feature_names)

# ==== 12. 聚类分析（KMeans）====
num_clusters = 5
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=list(basic_stopwords))
X = tfidf_vectorizer.fit_transform(texts_for_vector)
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans.fit(X)
df['cluster'] = kmeans.labels_

# ==== 13. 聚类可视化（降维）====
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())
plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=df['cluster'], cmap='rainbow', alpha=0.6)
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.title('Text Clusters Visualization')
plt.show()

# ==== 14. 人工补充情感词典建议与导出（未命中情感词）====
missed_words = []
for tokens in tokenized_texts:
    missed_words += [w for w in tokens if w not in sentiment_dict]
missed_freq = Counter(missed_words)
top50_missed = missed_freq.most_common(50)

if missed_freq:
    missed_df = pd.DataFrame(missed_freq.most_common(), columns=["word", "count"])
    missed_df.to_csv('missed_sentiment_words.csv', index=False, encoding='utf-8-sig')
    print("\n已导出未命中高频词至 missed_sentiment_words.csv，人工补充情感分数后可追加至原词典。")
else:
    print("未命中高频词为空，未生成csv文件。")

# ==== 15. 主要中性词统计与保存(txt) ====
neutral_words = []
for tokens in tokenized_texts:
    neutral_words += [w for w in tokens if sentiment_dict.get(w, None) == 0]
neutral_freq = Counter(neutral_words)
top50_neutral = neutral_freq.most_common(50)

if neutral_freq:
    with open('neutral_words.txt', 'w', encoding='utf-8') as f:
        for w, c in neutral_freq.most_common():
            f.write(f"{w}\t{c}\n")
    print("\n主要中性词已保存至 neutral_words.txt，可进行人工赋值。")
else:
    print("未检出中性词，未生成txt文件。")

# ==== 16. 输出主要分析结果为文本报告 ====
report_lines = []
report_lines.append("【情感词典分析自动报告】")
report_lines.append("1. 数据总览")
report_lines.append(f"  - 评论总数: {len(df)}")
report_lines.append(f"  - 平均每条评论命中情感词个数: {df['sentiment_hit'].mean():.2f}")
report_lines.append(f"  - 情感分数均值: {df['sentiment'].mean():.2f}")
report_lines.append(f"  - 情感分数中位数: {df['sentiment'].median():.2f}")
report_lines.append(f"  - 情感分数标准差: {df['sentiment'].std():.2f}")
report_lines.append("")

report_lines.append("2. 词频统计（前20）")
for w, c in top20_words:
    report_lines.append(f"  - {w}: {c}")
report_lines.append("")

report_lines.append("3. 共现网络")
report_lines.append(f"  - 共现网络节点数: {cooccur_node_count}")
report_lines.append(f"  - 共现网络边数: {cooccur_edge_count}")
if cooccur_node_count > 0:
    report_lines.append(f"  - 节点前10度中心性（最活跃词）:")
    node_degrees = sorted(G.degree, key=lambda x: x[1], reverse=True)[:10]
    for n, deg in node_degrees:
        report_lines.append(f"    - {n}: {deg}")
report_lines.append("")

report_lines.append("4. 主题建模（LDA）前5主题关键词")
for topic_idx, words in lda_topics:
    report_lines.append(f"  - Topic #{topic_idx+1}: {' '.join(words)}")
report_lines.append("")

report_lines.append("5. KMeans聚类")
report_lines.append(f"  - 聚类簇数量: {num_clusters}")
cluster_counts = df['cluster'].value_counts().to_dict()
for k in sorted(cluster_counts.keys()):
    report_lines.append(f"    - Cluster {k}: {cluster_counts[k]} 条评论")
report_lines.append("")

report_lines.append("6. 人工补充建议（未命中前50高频词）")
if top50_missed:
    for w, c in top50_missed:
        report_lines.append(f"  - {w}: {c}")
else:
    report_lines.append("  - 未命中高频词为空。")
report_lines.append("")

report_lines.append("7. 主要中性词（前50）")
if top50_neutral:
    for w, c in top50_neutral:
        report_lines.append(f"  - {w}: {c}")
else:
    report_lines.append("  - 未检出中性词。")
report_lines.append("")

report_content = '\n'.join(report_lines)
with open('analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_content)

print("\n已输出完整分析报告至 analysis_report.txt。")