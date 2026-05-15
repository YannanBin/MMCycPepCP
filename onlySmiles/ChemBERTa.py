from datetime import datetime

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from evaluate import compute_metrics
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers.trainer_utils import EvalPrediction
from sklearn.model_selection import KFold

# SMILES 数据增强
def augment_smiles(smiles: str) -> str:
    """
    通过 RDKit 随机重排 SMILES 字符串，生成等价表示
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        new_smiles = Chem.MolToSmiles(mol, doRandom=True)
        return new_smiles
    except:
        return smiles


# 自定义数据集类
class CyclicPeptideDataset(Dataset):
    def __init__(self, smiles, labels, tokenizer, max_length=512, augment=False):
        self.smiles = smiles
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smile = str(self.smiles[idx])
        if self.augment:
            smile = augment_smiles(smile)
        encoding = self.tokenizer(
            smile,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(self.labels[idx])
        }
        return item


# 加载和预处理数据
def load_data(file_path: str):
    df = pd.read_csv(file_path)
    label_columns = [col for col in df.columns if col not in ['CPKB ID', 'SMILES']]
    if not label_columns:
        raise ValueError("没有找到有效的标签列，请检查 CSV 文件结构")
    smiles = df['SMILES'].values
    labels = df[label_columns].values.astype(float)
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("标签列包含非二元值（0 或 1）")

    # 过采样正样本
    # pos_indices = np.where(labels.sum(axis=1) > 0)[0]
    # smiles = np.concatenate([smiles, smiles[pos_indices]])
    # labels = np.concatenate([labels, labels[pos_indices]])

    print("标签分布（正样本比例）：")
    sparse_labels = []
    for i, col in enumerate(label_columns):
        pos_ratio = np.mean(labels[:, i])
        print(f"  {col}: {pos_ratio:.4f}")
    #     if pos_ratio < 0.01:
    #         sparse_labels.append(col)
    # if sparse_labels:
    #     print(f"稀疏标签（正样本比例 <0.01）: {sparse_labels}")
    invalid_smiles = [s for s in smiles if Chem.MolFromSmiles(str(s)) is None]
    print(f"无效 SMILES 数量: {len(invalid_smiles)}")
    print(f"样本总数: {len(smiles)}")
    return smiles, labels, label_columns


# 计算标签权重以处理不平衡
def compute_label_weights(labels):
    n_samples = labels.shape[0]
    n_labels = labels.shape[1]
    pos_weights = []
    for i in range(n_labels):
        pos_count = np.sum(labels[:, i])
        neg_count = n_samples - pos_count
        weight = 1.0 if pos_count == 0 or neg_count == 0 else min(neg_count / pos_count, 5.0)
        pos_weights.append(weight)
    print(f"正样本权重: {pos_weights}")
    return torch.tensor(pos_weights, dtype=torch.float)


# 自定义评估函数
def evaluate(model, dataloader, device, pos_weights):
    """
    评估模型，返回指标、SMILES、真实标签和预测标签。

    Args:
        model: 评估模型
        dataloader: 数据加载器
        device: 设备 (cuda/cpu)
        pos_weights: 标签权重
        smiles: SMILES 列表（可选，用于保存结果）

    Returns:
        metrics: 评估指标
        all_smiles: SMILES 列表（如果提供 smiles）
        all_labels: 真实标签
        predictions: 预测标签（阈值 0.5）
    """
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    # loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
    loss_fct = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            # print(f"Evaluate - Logits 形状: {logits.shape}, Labels 形状: {labels.shape}")
            loss = loss_fct(logits, labels)
            total_loss += loss.item()
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    eval_pred = EvalPrediction(predictions=all_logits, label_ids=all_labels)
    metrics = compute_metrics(eval_pred)
    metrics['eval_loss'] = avg_loss

    return metrics

def save_metrics(metrics, epoch=None, file_path='./eval_metrics.txt'):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_path, 'a') as f:
        if epoch is not None:
            f.write(f"\nEpoch {epoch} Metrics ({timestamp}):\n")
        else:
            f.write(f"\nFinal Test Set Metrics ({timestamp}):\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value:.4f}\n")


# 自定义训练函数
def train_model(
        model,
        train_loader,
        device,
        pos_weights,
        num_epochs=30,
        learning_rate=5e-5,
        weight_decay=0.2,
        save_path='./chemberta_multi_label_final'
):
    """
    自定义训练循环，使用测试集评估选择最佳模型。

    Args:
        model: 训练模型
        train_loader: 训练集 DataLoader
        device: 设备 (cuda/cpu)
        pos_weights: 标签权重
        num_epochs: 训练轮数
        learning_rate: 学习率
        weight_decay: 权重衰减
        save_path: 模型保存路径
    """
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    # loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
    loss_fct = nn.BCEWithLogitsLoss()

    best_absolute_true = 0.0
    best_metrics = None

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            # labels = labels * 0.9 + 0.05  # 标签平滑
            outputs = model(**inputs)
            logits = outputs.logits  # 直接使用 logits，形状: [batch_size, num_labels]
            # print(f"Train - Logits 形状: {logits.shape}, Labels 形状: {labels.shape}")
            loss = loss_fct(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Average Train Loss: {avg_train_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        """
        # 在测试集上评估以选择最佳模型
        test_metrics = evaluate(model, test_loader, device, pos_weights)
        # print(f"Test Metrics at Epoch {epoch + 1}:")
        # for key, value in test_metrics.items():
        #     print(f"  {key}: {value:.4f}")

        if test_metrics['absolute_true'] > best_absolute_true:
            best_absolute_true = test_metrics['absolute_true']
            best_metrics = test_metrics
            model.save_pretrained(save_path)
            train_loader.dataset.tokenizer.save_pretrained(save_path)
            # print(f"Best model saved with test precision_micro: {best_absolute_true:.4f}")
        """

    return best_metrics


def main():
    file_path = '../datasets/lastfinal_cyclic_peptides.csv'
    model_name = "../pre_model/DeepChem/ChemBERTa-77M-MLM"
    # model_name = "../pre_model/seyonec/ChemBERTa-zinc-base-v1"
    # model_name = "../pre_model/seyonec/SMILES_tokenized_PubChem_shard00_160k"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 先加载数据以获取标签数量
    smiles, labels, label_columns = load_data(file_path)
    print(f"动态获取的标签列：{label_columns}")

    # 五折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_metrics = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(smiles)):
        print(f"\nFold {fold + 1}/5")
        train_smiles = smiles[train_idx]
        train_labels = labels[train_idx]
        test_smiles = smiles[test_idx]
        test_labels = labels[test_idx]

        # 初始化模型
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(label_columns),
            problem_type="multi_label_classification"
        )

        pos_weights = compute_label_weights(labels)

        # 创建数据集
        train_dataset = CyclicPeptideDataset(train_smiles, train_labels, tokenizer, max_length=512, augment=False)
        test_dataset = CyclicPeptideDataset(test_smiles, test_labels, tokenizer, max_length=512, augment=False)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # 训练模型
        train_model(
            model,
            train_loader,
            device,
            pos_weights,
            num_epochs=100,
            learning_rate=1e-4,
            weight_decay=0.2,
            save_path='./chemberta_zinc_multi_label_final'
        )
        test_metrics = evaluate(
            model, test_loader, device, pos_weights
        )
        all_metrics.append(test_metrics)
        print(f"Fold {fold + 1} Test Metrics: {test_metrics}")
        # for key, value in test_metrics.items():
        #     print(f"  {key}: {value:.4f}")

        # save_metrics(test_metrics)
        #
        # print(f"评估指标已保存至 ./eval_metrics.txt")
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    print("\nFive-Fold Average Metrics:")
    for key, value in avg_metrics.items():
        print(f"{key}: {value:.4f}")

    # 保存评估指标
    # save_metrics(avg_metrics)
    # print(f"评估指标已保存至 ./eval_metrics.txt")

if __name__ == '__main__':
    main()
