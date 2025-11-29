import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 1. 数据加载和预处理
def load():
    d = pd.read_excel('葡萄酒.xlsx')
    print(f"数据形状: {d.shape}")
    print(f"缺失值: {d.isnull().sum().sum()}")
    return d


def split_xy(d):
    X = d.iloc[:, :-1]
    y = d.iloc[:, -1]
    return X, y


def norm(X_train, X_test):
    s = StandardScaler()
    X_train_n = s.fit_transform(X_train)
    X_test_n = s.transform(X_test)
    return X_train_n, X_test_n, X_train, X_test


# 2. 获取所有K值的准确率
def get_acc(X_train, X_test, y_train, y_test, k_list, metric):
    s = StandardScaler()
    X_train_n = s.fit_transform(X_train)
    X_test_n = s.transform(X_test)

    acc_list = []
    for k in k_list:
        clf = KNeighborsClassifier(n_neighbors=k, metric=metric)
        clf.fit(X_train_n, y_train)
        pred = clf.predict(X_test_n)
        acc = accuracy_score(y_test, pred)
        acc_list.append(acc)
    return acc_list


# 3. 图1: 欧氏距离K值影响
def fig1(k_list, acc_e):
    plt.figure(figsize=(10, 6))
    plt.plot(k_list, acc_e, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('K值', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.title('图1: 欧氏距离下K值对准确率的影响', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_list)
    plt.tight_layout()
    plt.savefig('图1_欧氏距离K值.png', dpi=300, bbox_inches='tight')
    plt.show()

# 4. 图2: 曼哈顿距离K值影响
def fig2(k_list, acc_m):
    plt.figure(figsize=(10, 6))
    plt.plot(k_list, acc_m, 'rs-', linewidth=2, markersize=8)
    plt.xlabel('K值', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.title('图2: 曼哈顿距离下K值对准确率的影响', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_list)
    plt.tight_layout()
    plt.savefig('图2_曼哈顿距离K值.png', dpi=300, bbox_inches='tight')
    plt.show()


# 5. 图3: 欧氏vs曼哈顿对比
def fig3(k_list, acc_e, acc_m):
    plt.figure(figsize=(10, 6))
    plt.plot(k_list, acc_e, 'bo-', label='欧氏距离', linewidth=2, markersize=8)
    plt.plot(k_list, acc_m, 'rs-', label='曼哈顿距离', linewidth=2, markersize=8)
    plt.xlabel('K值', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.title('图3: 欧氏距离 vs 曼哈顿距离对比', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(k_list)
    plt.tight_layout()
    plt.savefig('图3_距离度量对比.png', dpi=300, bbox_inches='tight')
    plt.show()


# 6. 图4: 最优参数混淆矩阵
def fig4(X_train_n, X_test_n, y_train, y_test, best_k):
    clf = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
    clf.fit(X_train_n, y_train)
    pred = clf.predict(X_test_n)
    cm = confusion_matrix(y_test, pred)

    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                square=True, annot_kws={'size': 14})
    plt.xlabel('预测类别', fontsize=12)
    plt.ylabel('真实类别', fontsize=12)
    plt.title(f'图4: 最优参数混淆矩阵 (K={best_k}, 欧氏距离)',
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('图4_混淆矩阵.png', dpi=300, bbox_inches='tight')
    plt.show()


# 7. 图5: 精度热力图
def fig5(k_list, acc_e, acc_m):
    acc_matrix = np.array([acc_e, acc_m])

    plt.figure(figsize=(12, 5))
    sns.heatmap(acc_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                cbar_kws={'label': '准确率'}, vmin=0.9, vmax=1.0,
                xticklabels=k_list, yticklabels=['欧氏距离', '曼哈顿距离'],
                annot_kws={'size': 10})
    plt.xlabel('K值', fontsize=12)
    plt.ylabel('距离度量', fontsize=12)
    plt.title('图5: 不同K值与距离度量的精度热力图', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('图5_精度热力图.png', dpi=300, bbox_inches='tight')
    plt.show()


# 8. 图6: 标准化效果对比
def fig6(X_train, X_test, y_train, y_test, k_list):
    # 标准化数据
    s = StandardScaler()
    X_train_n = s.fit_transform(X_train)
    X_test_n = s.transform(X_test)

    acc_norm = []
    acc_raw = []

    for k in k_list:
        # 标准化
        clf_n = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        clf_n.fit(X_train_n, y_train)
        acc_norm.append(accuracy_score(y_test, clf_n.predict(X_test_n)))

        # 未标准化
        clf_r = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        clf_r.fit(X_train, y_train)
        acc_raw.append(accuracy_score(y_test, clf_r.predict(X_test)))

    plt.figure(figsize=(10, 6))
    plt.plot(k_list, acc_norm, 'go-', label='标准化', linewidth=2, markersize=8)
    plt.plot(k_list, acc_raw, 'mo--', label='未标准化', linewidth=2, markersize=8)
    plt.xlabel('K值', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.title('图6: 标准化与未标准化对模型精度的影响', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(k_list)
    plt.tight_layout()
    plt.savefig('图6_标准化对比.png', dpi=300, bbox_inches='tight')
    plt.show()


# 主程序
if __name__ == '__main__':
    # 加载数据
    d = load()
    X, y = split_xy(d)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 标准化
    s = StandardScaler()
    X_train_n = s.fit_transform(X_train)
    X_test_n = s.transform(X_test)

    # K值列表
    k_list = list(range(1, 16))

    # 计算准确率
    print("计算准确率中...")
    acc_e = get_acc(X_train, X_test, y_train, y_test, k_list, 'euclidean')
    acc_m = get_acc(X_train, X_test, y_train, y_test, k_list, 'manhattan')

    # 打印结果
    print("\n=== 实验结果 ===")
    print(f"欧氏距离: {[f'{a:.3f}' for a in acc_e]}")
    print(f"曼哈顿距离: {[f'{a:.3f}' for a in acc_m]}")

    best_k_e = k_list[np.argmax(acc_e)]
    best_k_m = k_list[np.argmax(acc_m)]
    print(f"\n最优K(欧氏): {best_k_e}, 准确率: {max(acc_e):.3f}")
    print(f"最优K(曼哈顿): {best_k_m}, 准确率: {max(acc_m):.3f}")

    # 逐个输出图表
    print("\n输出图表中...\n")
    fig1(k_list, acc_e)
    fig2(k_list, acc_m)
    fig3(k_list, acc_e, acc_m)
    fig4(X_train_n, X_test_n, y_train, y_test, best_k_e)
    fig5(k_list, acc_e, acc_m)
    fig6(X_train, X_test, y_train, y_test, k_list)

    print("所有图表已保存！")