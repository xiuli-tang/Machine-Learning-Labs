import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA

# 创建图片保存目录
if not os.path.exists("img"):
    os.makedirs("img")
# 读取文件
def load(path):
    df = pd.read_csv(path, encoding="latin1")
    df = df.iloc[:, :2]
    df.columns = ["y", "x"]
    return df
# 文本清洗
def prep(s):
    s = s.str.lower()
    s = s.str.replace(r"[^a-z0-9 ]", " ", regex=True)
    return s
# 统计 top20 词
def top20(df, lab):
    txt = " ".join(df[df["y"] == lab]["x"])
    words = txt.split()
    c = pd.Series(words).value_counts()[:20]
    return c
# 词云
def wc(txt, title, name):
    w = WordCloud(width=900, height=400, background_color="white").generate(txt)
    plt.figure(figsize=(8, 4))
    plt.imshow(w, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.savefig(f"img/{name}.png", dpi=300)
    plt.show()
# 模型训练
def train(x, y):
    m = MultinomialNB()
    m.fit(x, y)
    return m
def main():
    df = load("archive/spam.csv")
    df["x"] = prep(df["x"])
    # 类别占比饼图
    plt.figure(figsize=(5, 5))
    df["y"].value_counts().plot.pie(autopct="%1.1f%%")
    plt.title("Label Distribution (Ham vs Spam)")
    plt.ylabel("")
    plt.savefig("img/pie.png", dpi=300)
    plt.show()
    # Top20 词频
    c1 = top20(df, "ham")
    c2 = top20(df, "spam")
    plt.figure(figsize=(10, 4))
    sns.barplot(x=c1.values, y=c1.index)
    plt.title("Top20 Ham Words")
    plt.savefig("img/top_ham.png", dpi=300)
    plt.show()
    plt.figure(figsize=(10, 4))
    sns.barplot(x=c2.values, y=c2.index)
    plt.title("Top20 Spam Words")
    plt.savefig("img/top_spam.png", dpi=300)
    plt.show()
    # 词云
    txt_h = " ".join(df[df["y"] == "ham"]["x"])
    txt_s = " ".join(df[df["y"] == "spam"]["x"])
    wc(txt_h, "WordCloud Ham", "wc_ham")
    wc(txt_s, "WordCloud Spam", "wc_spam")
    # TF-IDF
    vec = TfidfVectorizer(min_df=2)
    x = vec.fit_transform(df["x"])
    y = df["y"]
    # TF-IDF 最重要词（按平均 TF-IDF 排序）
    avg_tfidf = x.mean(axis=0).A1
    idx = avg_tfidf.argsort()[-20:][::-1]
    feats = [vec.get_feature_names_out()[i] for i in idx]
    plt.figure(figsize=(10, 4))
    sns.barplot(x=avg_tfidf[idx], y=feats)
    plt.title("Top TF-IDF Features")
    plt.savefig("img/tfidf.png", dpi=300)
    plt.show()
    # PCA 降维
    p = PCA(n_components=2)
    pts = p.fit_transform(x.toarray())
    df_pca = pd.DataFrame({"p1": pts[:, 0], "p2": pts[:, 1], "y": y})
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df_pca, x="p1", y="p2", hue="y", s=10)
    plt.title("PCA Visualization")
    plt.savefig("img/pca.png", dpi=300)
    plt.show()
    # 划分数据
    x_train, x_test, y_train, y_test = train_test_split(
        df["x"], y, test_size=0.2, random_state=42
    )
    x_train = vec.fit_transform(x_train)
    x_test = vec.transform(x_test)
    m = train(x_train, y_train)
    y_pred = m.predict(x_test)
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    plt.title("Confusion Matrix")
    plt.savefig("img/cm.png", dpi=300)
    plt.show()
    # ROC 曲线
    y_prob = m.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test.map({"ham": 0, "spam": 1}), y_prob)
    r = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "--")
    plt.title(f"ROC Curve AUC={r:.3f}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.savefig("img/roc.png", dpi=300)
    plt.show()
if __name__ == "__main__":
    main()
