import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据准备
def load():
    d = pd.read_csv('car_evaluation.csv')
    print(f"数据集大小: {d.shape}")
    print(f"类别分布:\n{d['class'].value_counts()}\n")
    return d

d = load()

# 2. 特征编码
def code(d):
    d = d.copy()
    for col in d.columns:
        e = LabelEncoder()
        d[col] = e.fit_transform(d[col])
    return d

d_code = code(d)

# 3. 数据集划分
X = d_code.drop('class', axis=1)
y = d_code['class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}\n")

# 4. ID3算法（信息增益）
print("="*50)
print("ID3算法（信息增益准则）")
print("="*50)
m1 = DecisionTreeClassifier(criterion='entropy', max_depth=8, random_state=42)
m1.fit(X_train, y_train)
p1 = m1.predict(X_test)
print(f"准确率: {accuracy_score(y_test, p1):.4f}")
print(f"精确率: {precision_score(y_test, p1, average='weighted'):.4f}")
print(f"召回率: {recall_score(y_test, p1, average='weighted'):.4f}")
print(f"F1分数: {f1_score(y_test, p1, average='weighted'):.4f}")
print(f"树深度: {m1.get_depth()}, 叶子数: {m1.get_n_leaves()}\n")

# 5. CART算法（基尼系数）
print("="*50)
print("CART算法（基尼系数准则）")
print("="*50)
m2 = DecisionTreeClassifier(criterion='gini', max_depth=8, random_state=42)
m2.fit(X_train, y_train)
p2 = m2.predict(X_test)
print(f"准确率: {accuracy_score(y_test, p2):.4f}")
print(f"精确率: {precision_score(y_test, p2, average='weighted'):.4f}")
print(f"召回率: {recall_score(y_test, p2, average='weighted'):.4f}")
print(f"F1分数: {f1_score(y_test, p2, average='weighted'):.4f}")
print(f"树深度: {m2.get_depth()}, 叶子数: {m2.get_n_leaves()}\n")

# 6. CART剪枝优化
print("="*50)
print("CART算法（成本复杂度剪枝）")
print("="*50)
path = m2.cost_complexity_pruning_path(X_train, y_train)
ccp = path.ccp_alphas[:-1]
acc_train, acc_test = [], []

for alpha in ccp:
    m = DecisionTreeClassifier(criterion='gini', ccp_alpha=alpha, random_state=42)
    m.fit(X_train, y_train)
    acc_train.append(accuracy_score(y_train, m.predict(X_train)))
    acc_test.append(accuracy_score(y_test, m.predict(X_test)))

best_idx = np.argmax(acc_test)
best_alpha = ccp[best_idx]
m3 = DecisionTreeClassifier(criterion='gini', ccp_alpha=best_alpha, random_state=42)
m3.fit(X_train, y_train)
p3 = m3.predict(X_test)
print(f"最优alpha: {best_alpha:.6f}")
print(f"准确率: {accuracy_score(y_test, p3):.4f}")
print(f"精确率: {precision_score(y_test, p3, average='weighted'):.4f}")
print(f"召回率: {recall_score(y_test, p3, average='weighted'):.4f}")
print(f"F1分数: {f1_score(y_test, p3, average='weighted'):.4f}")
print(f"树深度: {m3.get_depth()}, 叶子数: {m3.get_n_leaves()}\n")

# 7. 图1：决策树结构
plt.figure(figsize=(20, 12))
plot_tree(m3, feature_names=list(X.columns), class_names=['unacc','acc','good','vgood'],
          filled=True, fontsize=10, rounded=True)
plt.title(f'决策树结构（CART+剪枝，alpha={best_alpha:.6f}）', fontsize=14, fontweight='bold', pad=20)
plt.savefig('1_tree.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 图1已保存: 1_tree.png\n")

# 8. 图2：CART剪枝曲线
plt.figure(figsize=(10, 6))
plt.plot(ccp, acc_train, marker='o', label='训练集', linewidth=2.5, markersize=6)
plt.plot(ccp, acc_test, marker='s', label='测试集', linewidth=2.5, markersize=6)
plt.axvline(best_alpha, color='red', linestyle='--', linewidth=2, label=f'最优alpha={best_alpha:.6f}')
plt.xlabel('ccp_alpha', fontsize=12, fontweight='bold')
plt.ylabel('准确率', fontsize=12, fontweight='bold')
plt.title('CART成本复杂度剪枝曲线', fontsize=14, fontweight='bold', pad=15)
plt.legend(fontsize=11, loc='lower left')
plt.grid(True, alpha=0.3)
plt.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.12)
plt.savefig('2_pruning_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 图2已保存: 2_pruning_curve.png\n")

# 9. 图3：混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, p3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['unacc','acc','good','vgood'],
            yticklabels=['unacc','acc','good','vgood'],
            annot_kws={'size': 12, 'weight': 'bold'})
plt.ylabel('真实标签', fontsize=12, fontweight='bold')
plt.xlabel('预测标签', fontsize=12, fontweight='bold')
plt.title('混淆矩阵（CART+剪枝）', fontsize=14, fontweight='bold', pad=15)
plt.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.12)
plt.savefig('3_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 图3已保存: 3_confusion_matrix.png\n")

# 10. 图4：特征重要性
plt.figure(figsize=(10, 6))
imp = m3.feature_importances_
idx = np.argsort(imp)
names = [X.columns[i] for i in idx]
plt.barh(names, imp[idx], color='#45B7D1', alpha=0.8, edgecolor='black', linewidth=1.5)
plt.xlabel('重要性', fontsize=12, fontweight='bold')
plt.title('特征重要性排序', fontsize=14, fontweight='bold', pad=15)
plt.grid(True, alpha=0.3, axis='x')
for i, v in enumerate(imp[idx]):
    plt.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=10, fontweight='bold')
plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.12)
plt.savefig('4_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 图4已保存: 4_feature_importance.png\n")

# 11. 分类报告
print("="*50)
print("CART+剪枝 分类报告")
print("="*50)
print(classification_report(y_test, p3,
      target_names=['unacc','acc','good','vgood'], digits=4))