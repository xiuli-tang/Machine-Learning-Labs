import numpy as np
import matplotlib.pyplot as plt
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei','SimSun']
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
# ==================== 数据准备 ====================
d = load_iris()
X, y = d.data, d.target
# 标准化
s = StandardScaler()
X = s.fit_transform(X)
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
# ==================== KNN 超参数探索 ====================
def knn_k():
    k_range = range(1, 16)
    train_acc, test_acc = [], []
    for k in k_range:
        m = KNeighborsClassifier(n_neighbors=k)
        m.fit(X_train, y_train)
        train_acc.append(m.score(X_train, y_train))
        test_acc.append(m.score(X_test, y_test))

    return k_range, train_acc, test_acc
# ==================== 决策树超参数探索 ====================
def tree_depth():
    d_range = range(1, 11)
    train_acc, test_acc = [], []
    for d in d_range:
        m = DecisionTreeClassifier(max_depth=d, random_state=42)
        m.fit(X_train, y_train)
        train_acc.append(m.score(X_train, y_train))
        test_acc.append(m.score(X_test, y_test))
    return d_range, train_acc, test_acc
# ==================== 神经网络超参数探索 ====================
def nn_hidden_layer():
    layer_range = [(32,), (64,), (128,), (32, 32), (64, 32), (128, 64), (128, 64, 32)]
    train_acc, test_acc = [], []
    for hidden_layer in layer_range:
        m = MLPClassifier(hidden_layer_sizes=hidden_layer, max_iter=500, random_state=42)
        m.fit(X_train, y_train)
        train_acc.append(m.score(X_train, y_train))
        test_acc.append(m.score(X_test, y_test))
    return [str(h) for h in layer_range], train_acc, test_acc
# ==================== K-Fold 交叉验证 ====================
def cv_compare():
    knn = KNeighborsClassifier(n_neighbors=5)
    tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    nn = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    # KNN
    knn_5 = cross_val_score(knn, X_train, y_train, cv=5).mean()
    knn_10 = cross_val_score(knn, X_train, y_train, cv=10).mean()
    # 决策树
    tree_5 = cross_val_score(tree, X_train, y_train, cv=5).mean()
    tree_10 = cross_val_score(tree, X_train, y_train, cv=10).mean()
    # 神经网络
    nn_5 = cross_val_score(nn, X_train, y_train, cv=5).mean()
    nn_10 = cross_val_score(nn, X_train, y_train, cv=10).mean()
    return {
        'KNN-5Fold': knn_5,
        'KNN-10Fold': knn_10,
        'Tree-5Fold': tree_5,
        'Tree-10Fold': tree_10,
        'NN-5Fold': nn_5,
        'NN-10Fold': nn_10
    }
# ==================== 学习曲线 ====================
def lc(model, name):
    train_sz, train_s, val_s = learning_curve(
        model, X_train, y_train, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    return train_sz, train_s.mean(axis=1), val_s.mean(axis=1)
# ==================== 绘图 ====================
# 图1: KNN - K vs 准确率
k, train_k, test_k = knn_k()
plt.figure(figsize=(8, 5))
plt.plot(k, train_k, 'o-', label='Train', color='blue')
plt.plot(k, test_k, 's-', label='Test', color='red')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title('图1: KNN - K值 vs 准确率')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('图1.png', dpi=600, bbox_inches='tight')
plt.close()
# 图2: 决策树 - 深度 vs 准确率
d, train_d, test_d = tree_depth()
plt.figure(figsize=(8, 5))
plt.plot(d, train_d, 'o-', label='Train', color='blue')
plt.plot(d, test_d, 's-', label='Test', color='red')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('图2: 决策树 - 深度 vs 准确率')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('图2.png', dpi=600, bbox_inches='tight')
plt.close()
# 图3: 神经网络 - 隐层结构 vs 准确率
nn_layers, train_nn, test_nn = nn_hidden_layer()
plt.figure(figsize=(10, 5))
x_pos = np.arange(len(nn_layers))
plt.plot(x_pos, train_nn, 'o-', label='Train', color='blue')
plt.plot(x_pos, test_nn, 's-', label='Test', color='red')
plt.xlabel('Hidden Layer Sizes')
plt.ylabel('Accuracy')
plt.title('图3: 神经网络 - 隐层结构 vs 准确率')
plt.xticks(x_pos, nn_layers, rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('图3.png', dpi=600, bbox_inches='tight')
plt.close()
# 图4: K-Fold 对比（包括三个模型）
cv_r = cv_compare()
methods = list(cv_r.keys())
scores = list(cv_r.values())
colors = ['blue', 'blue', 'green', 'green', 'red', 'red']
plt.figure(figsize=(10, 5))
plt.bar(methods, scores, color=colors, alpha=0.7)
plt.ylabel('Cross-Val Score')
plt.title('图4: 三模型 5-Fold vs 10-Fold 交叉验证对比')
plt.ylim([0.85, 1.0])
for i, v in enumerate(scores):
    plt.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('图4.png', dpi=600, bbox_inches='tight')
plt.close()
# 图5: KNN 学习曲线
knn_m = KNeighborsClassifier(n_neighbors=5)
sz, tr, val = lc(knn_m, 'KNN')
plt.figure(figsize=(8, 5))
plt.plot(sz, tr, 'o-', label='Train', color='blue')
plt.plot(sz, val, 's-', label='Val', color='red')
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.title('图5: KNN 学习曲线')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('图5.png', dpi=600, bbox_inches='tight')
plt.close()
# 图6: 决策树学习曲线
tree_m = DecisionTreeClassifier(max_depth=5, random_state=42)
sz, tr, val = lc(tree_m, 'Tree')
plt.figure(figsize=(8, 5))
plt.plot(sz, tr, 'o-', label='Train', color='blue')
plt.plot(sz, val, 's-', label='Val', color='red')
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.title('图6: 决策树学习曲线')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('图6.png', dpi=600, bbox_inches='tight')
plt.close()
# 图7: 神经网络学习曲线
nn_m = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
sz, tr, val = lc(nn_m, 'NN')
plt.figure(figsize=(8, 5))
plt.plot(sz, tr, 'o-', label='Train', color='blue')
plt.plot(sz, val, 's-', label='Val', color='red')
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.title('图7: 神经网络学习曲线')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('图7.png', dpi=600, bbox_inches='tight')
plt.close()
# 图8: 三个模型最优性能对比
best_k_idx = np.argmax(test_k)
best_d_idx = np.argmax(test_d)
best_nn_idx = np.argmax(test_nn)
knn_opt = KNeighborsClassifier(n_neighbors=5)
knn_opt.fit(X_train, y_train)
knn_best_acc = knn_opt.score(X_test, y_test)
tree_opt = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_opt.fit(X_train, y_train)
tree_best_acc = tree_opt.score(X_test, y_test)
nn_opt = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
nn_opt.fit(X_train, y_train)
nn_best_acc = nn_opt.score(X_test, y_test)
models = ['KNN', 'Decision Tree', 'Neural Network']
accuracies = [knn_best_acc, tree_best_acc, nn_best_acc]
colors_model = ['blue', 'green', 'red']
plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies, color=colors_model, alpha=0.7)
plt.ylabel('Test Accuracy')
plt.title('图8: 三个模型最优性能对比')
plt.ylim([0.9, 1.0])
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    plt.text(i, acc + 0.005, f'{acc:.4f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('图8.png', dpi=600, bbox_inches='tight')
plt.close()
# 图9: 混淆矩阵 - KNN
y_pred_knn = knn_opt.predict(X_test)
cm_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(7, 5))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('图9: KNN 混淆矩阵')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('图9.png', dpi=600, bbox_inches='tight')
plt.close()
# 图10: 混淆矩阵 - 决策树
y_pred_tree = tree_opt.predict(X_test)
cm_tree = confusion_matrix(y_test, y_pred_tree)
plt.figure(figsize=(7, 5))
sns.heatmap(cm_tree, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title('图10: 决策树 混淆矩阵')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('图10.png', dpi=600, bbox_inches='tight')
plt.close()
# 图11: 混淆矩阵 - 神经网络
y_pred_nn = nn_opt.predict(X_test)
cm_nn = confusion_matrix(y_test, y_pred_nn)
plt.figure(figsize=(7, 5))
sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Reds', cbar=False)
plt.title('图11: 神经网络 混淆矩阵')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('图11.png', dpi=600, bbox_inches='tight')
plt.close()
# ==================== 总结输出 ====================
print("\n=== 交叉验证结果 ===")
for method, score in cv_r.items():
    print(f"{method}: {score:.4f}")
print(f"\n最优 KNN K 值: {list(k)[best_k_idx]}, 测试准确率: {test_k[best_k_idx]:.4f}")
print(f"最优树深度: {list(d)[best_d_idx]}, 测试准确率: {test_d[best_d_idx]:.4f}")
print(f"最优NN隐层: {nn_layers[best_nn_idx]}, 测试准确率: {test_nn[best_nn_idx]:.4f}")
print(f"\n=== 三模型最终性能对比 ===")
print(f"KNN 测试准确率: {knn_best_acc:.4f}")
print(f"决策树 测试准确率: {tree_best_acc:.4f}")
print(f"神经网络 测试准确率: {nn_best_acc:.4f}")
best_model = models[np.argmax(accuracies)]
best_acc = max(accuracies)
print(f"\n最优模型: {best_model}, 准确率: {best_acc:.4f}")