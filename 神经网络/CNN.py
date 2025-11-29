import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = './archive/dataset-resized'
trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
trans_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
full = datasets.ImageFolder(root=data_dir, transform=trans)
n = len(full)
train_n = int(0.8 * n)
val_n = int(0.1 * n)
test_n = n - train_n - val_n
train_d, val_d, test_d = random_split(full, [train_n, val_n, test_n])
class CNN(nn.Module):
    def __init__(self, num_c):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_c)
        self.drop = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x
if __name__ == '__main__':
    print(f'设备: {device}')
    print(f'类别: {full.classes}')
    print(f'总数: {len(full)}')
    print(f'train: {len(train_d)}, val: {len(val_d)}, test: {len(test_d)}\n')
    batch = 32
    train_ldr = DataLoader(train_d, batch_size=batch, shuffle=True, num_workers=0)
    val_ldr = DataLoader(val_d, batch_size=batch, shuffle=False, num_workers=0)
    test_ldr = DataLoader(test_d, batch_size=batch, shuffle=False, num_workers=0)
    num_c = len(full.classes)
    model = models.resnet18(weights='DEFAULT')
    model.fc = nn.Linear(512, num_c)
    model = model.to(device)
    print(f'模型: ResNet18')
    print(f'参数: {sum(p.numel() for p in model.parameters()):,}\n')
    loss_fn = nn.CrossEntropyLoss()
    lr_list = [1e-5, 1e-4, 1e-3]
    results = {}
    for lr in lr_list:
        print(f'========== LR: {lr} ==========')
        opt = optim.Adam(model.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
        train_loss_his = []
        val_loss_his = []
        train_acc_his = []
        val_acc_his = []
        best_val_ac = 0
        patience = 10
        p_cnt = 0
        for ep in range(50):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            for img, lab in tqdm(train_ldr, desc=f'EP{ep}', leave=False):
                img, lab = img.to(device), lab.to(device)
                opt.zero_grad()
                out = model(img)
                loss = loss_fn(out, lab)
                loss.backward()
                opt.step()
                train_loss += loss.item()
                _, pred = out.max(1)
                train_correct += (pred == lab).sum().item()
                train_total += lab.size(0)
            train_loss_avg = train_loss / len(train_ldr)
            train_acc = train_correct / train_total
            train_loss_his.append(train_loss_avg)
            train_acc_his.append(train_acc)
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for img, lab in val_ldr:
                    img, lab = img.to(device), lab.to(device)
                    out = model(img)
                    loss = loss_fn(out, lab)
                    val_loss += loss.item()
                    _, pred = out.max(1)
                    val_correct += (pred == lab).sum().item()
                    val_total += lab.size(0)
            val_loss_avg = val_loss / len(val_ldr)
            val_acc = val_correct / val_total
            val_loss_his.append(val_loss_avg)
            val_acc_his.append(val_acc)
            print(f'EP{ep} | train_loss: {train_loss_avg:.4f}, train_acc: {train_acc:.4f} | val_loss: {val_loss_avg:.4f}, val_acc: {val_acc:.4f}')
            sch.step(val_loss_avg)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                p_cnt = 0
            else:
                p_cnt += 1
            if p_cnt >= patience:
                print(f'早停: {patience} epoch无改进\n')
                break
        results[lr] = {
            'train_loss': train_loss_his,
            'val_loss': val_loss_his,
            'train_acc': train_acc_his,
            'val_acc': val_acc_his,
            'best_val_acc': best_val_acc
        }
    print('========== 测试集评估 ==========')
    model.eval()
    test_correct = 0
    test_total = 0
    all_pred = []
    all_lab = []
    with torch.no_grad():
        for img, lab in test_ldr:
            img, lab = img.to(device), lab.to(device)
            out = model(img)
            _, pred = out.max(1)
            test_correct += (pred == lab).sum().item()
            test_total += lab.size(0)
            all_pred.extend(pred.cpu().numpy())
            all_lab.extend(lab.cpu().numpy())
    test_acc = test_correct / test_total
    print(f'测试准确率: {test_acc:.4f}\n')
    print('分类报告:')
    print(classification_report(all_lab, all_pred, target_names=full.classes))
    cm = confusion_matrix(all_lab, all_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=full.classes, yticklabels=full.classes)
    plt.title(f'混淆矩阵 (Acc: {test_acc:.4f})')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print('混淆矩阵已保存\n')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for lr in lr_list:
        r = results[lr]
        axes[0].plot(r['train_loss'], label=f'LR={lr} train')
        axes[0].plot(r['val_loss'], label=f'LR={lr} val')
        axes[1].plot(r['train_acc'], label=f'LR={lr} train')
        axes[1].plot(r['val_acc'], label=f'LR={lr} val')
    axes[0].set_title('损失曲线')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title('准确率曲线')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
    print('学习曲线已保存\n')
    print('========== 学习率对比总结 ==========')
    for lr in lr_list:
        best_acc = results[lr]['best_val_acc']
        print(f'LR {lr}: 最高val_acc = {best_acc:.4f}')
    print('\n✓ 项目完成！')