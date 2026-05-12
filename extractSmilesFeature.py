import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
from typing import Union, Tuple, Any
import os


# 自定义数据集类
class SMILESDataset(Dataset):
    def __init__(self, smiles, cpkb_ids, tokenizer, max_length=512):
        self.smiles = smiles
        self.cpkb_ids = cpkb_ids
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smile = str(self.smiles[idx])
        cpkb_id = str(self.cpkb_ids[idx])
        encoding = self.tokenizer(
            smile,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'cpkb_id': cpkb_id
        }


# 加载和预处理数据
def load_data(file_path: str):
    df = pd.read_csv(file_path)
    if 'SMILES' not in df.columns or 'CPKB ID' not in df.columns:
        raise ValueError("CSV 文件中未找到 'SMILES' 或 'CPKB ID' 列")
    smiles = df['SMILES'].values
    cpkb_ids = df['CPKB ID'].values
    invalid_smiles = [s for s in smiles if Chem.MolFromSmiles(str(s)) is None]
    print(f"无效 SMILES 数量: {len(invalid_smiles)}")
    print(f"样本总数: {len(smiles)}")
    return smiles, cpkb_ids


# 提取特征并保存
def extract_features(
        model,
        dataloader,
        device,
        output_path: str = './cpkb_features'
):
    model.eval()
    os.makedirs(output_path, exist_ok=True)
    processed_count = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            cpkb_ids = batch['cpkb_id']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # 使用 pooler_output 作为特征，形状: [batch_size, hidden_size]
            features = outputs.pooler_output.cpu().numpy()

            # 为每个样本保存单独的 .npy 文件
            for cpkb_id, feature in zip(cpkb_ids, features):
                # 清理 CPKB ID 中的非法字符以确保文件名有效
                safe_cpkb_id = cpkb_id.replace('/', '_').replace('\\', '_')
                feature_file = os.path.join(output_path, f"{safe_cpkb_id}.npy")
                np.save(feature_file, feature)
                processed_count += 1

            print(f"已处理 {processed_count} 个样本", end='\r')

    print(f"\n共保存 {processed_count} 个特征文件至: {output_path}")


def main():
    # 参数设置
    file_path = 'datasets/lastfinal_cyclic_peptides.csv'
    model_name = "pre_model/DeepChem/ChemBERTa-77M-MLM"
    output_dir = 'features/smiles_features'

    # 初始化 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"使用设备: {device}")

    # 加载数据
    smiles, cpkb_ids = load_data(file_path)

    # 创建数据集和 DataLoader
    dataset = SMILESDataset(smiles, cpkb_ids, tokenizer, max_length=512)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    # 提取并保存特征
    extract_features(
        model,
        dataloader,
        device,
        output_path=output_dir
    )


if __name__ == '__main__':
    main()