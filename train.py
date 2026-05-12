import os
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from loss import compute_label_weights, FocalLoss, AsymmetricLoss, ClassBalancedFocalLoss, get_loss
import torch

from sampling import select_hard_negatives, select_hard_positives
from utils import custom_collate_fn


def train_model(model, train_loader, config, device, dataset, fold):
    optimizer = AdamW(model.parameters(), lr=config['train']['learning_rate'])
    # optimizer = AdamW(model.parameters(), lr=config['train']['learning_rate'], weight_decay=config['train']['weight_decay'])
    # 使用余弦退火学习率调度器
    total_steps = len(train_loader) * config['train']['epochs']
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # # 计算标签权重
    # pos_weights = None
    # if config['train']['use_label_weights']:
    #     dataset_path = config['data']['data_path']
    #     df = pd.read_csv(dataset_path)
    #     label_cols = [col for col in df.columns if col not in ['CPKB ID', 'SMILES']]
    #     labels = df[label_cols].values.astype(np.float32)
    #     pos_weights = compute_label_weights(labels).to(device)
    #
    # # 初始化损失函数
    # loss_type = config['train']['loss_type']
    # if loss_type == 'focal loss':
    #     criterion = FocalLoss(
    #         alpha=config['train']['focal_alpha'],
    #         gamma=config['train']['focal_gamma'],
    #         pos_weights=pos_weights
    #     )
    # elif loss_type == 'cb focal loss':
    #     # 统计每类正样本数（labels 是 NumPy 格式）
    #     dataset_path = config['data']['data_path']
    #     df = pd.read_csv(dataset_path)
    #     label_cols = [col for col in df.columns if col not in ['CPKB ID', 'SMILES']]
    #     labels = df[label_cols].values.astype(np.float32)
    #     samples_per_class = torch.tensor(labels.sum(axis=0), dtype=torch.float)
    #
    #     criterion = ClassBalancedFocalLoss(
    #         beta=config['train']['cb_beta'],
    #         gamma=config['train']['cb_gamma'],
    #         alpha=config['train']['cb_alpha'],
    #         samples_per_class=samples_per_class
    #     )
    #
    # elif loss_type == 'asymmetric loss':
    #     criterion = AsymmetricLoss(
    #         gamma_neg=config['train']['asl_gamma_neg'],
    #         gamma_pos=config['train']['asl_gamma_pos'],
    #         pos_weights=pos_weights
    #     )
    # else:
    #     criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights if pos_weights is not None else None)
    #
    # criterion.to(device)

    # === 初始化损失函数 ===
    dataset_path = config['data']['data_path']
    df = pd.read_csv(dataset_path)
    label_cols = [col for col in df.columns if col not in ['CPKB ID', 'SMILES']]
    labels = df[label_cols].values.astype(np.float32)

    criterion = get_loss(config, labels=labels).to(device)

    # Ensure model save directory exists
    model_save_dir = os.path.dirname(config['train']['model_save_path'])
    os.makedirs(model_save_dir, exist_ok=True)

    # Define fold-specific model save path
    model_save_path = os.path.join(model_save_dir, f'model_fold_{fold}.pt')

    # 获取原始训练索引
    train_indices = torch.tensor(train_loader.dataset.indices, dtype=torch.long)

    best_loss = float('inf')

    for epoch in range(config['train']['epochs']):
        model.train()
        total_loss = 0

        # 添加-动态 困难样本采样（困难负样本 + 困难正样本）
        if epoch % config['train']['selected_epoch'] == 0 and epoch > 0:
            selected_indices = [train_indices]

            if config['train']['use_hard_negative']:
                hard_neg_local = select_hard_negatives(model, train_loader.dataset, config,
                                                       k=config['train']['n_samples'],
                                                       confidence_threshold=config['train'][
                                                           'confidence_threshold_negative'])
                if len(hard_neg_local) > 0:
                    hard_neg_global = torch.tensor(train_loader.dataset.indices)[hard_neg_local]
                    selected_indices.append(hard_neg_global)

            if config['train']['use_hard_positive']:
                hard_pos_local = select_hard_positives(model, train_loader.dataset, config,
                                                       k=config['train']['n_samples'],
                                                       confidence_threshold=config['train'][
                                                           'confidence_threshold_positive'])
                if len(hard_pos_local) > 0:
                    hard_pos_global = torch.tensor(train_loader.dataset.indices)[hard_pos_local]
                    selected_indices.append(hard_pos_global)

            combined_indices = torch.cat(selected_indices)
            combined_indices = combined_indices.cpu().numpy()

            # train_subset = Subset(dataset, combined_indices)
            train_subset = Subset(train_loader.dataset.dataset, combined_indices)

            train_loader = DataLoader(train_subset, batch_size=config['train']['batch_size'],
                                      shuffle=True, collate_fn=custom_collate_fn,
                                      num_workers=config['train']['num_workers'])
            print(f"Epoch {epoch + 1}: Updated train_loader with {len(combined_indices)} samples.")


        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['train']['epochs']}"):
            input_ids, attention_mask, graph_data, morgan_fp, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            graph_data = graph_data.to(device)
            morgan_fp = morgan_fp.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            # logits, loss = model(input_ids, graph_data, morgan_fp, attention_mask=attention_mask,
            #                      labels=labels, gnn_features=gnn_features, cnn_features=cnn_features)
            # logits, loss = model(input_ids, graph_data, morgan_fp, attention_mask=attention_mask,
            #                      labels=labels, gnn_features=gnn_features, cnn_features=cnn_features,
            #                      pos_weights=pos_weights)
            logits = model(input_ids, graph_data, morgan_fp, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            total_loss += loss.item()

        # print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")
        # print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        avg_loss = total_loss / len(train_loader)
        print(f"Fold {fold + 1}, Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
        # Save model if this epoch has the best loss for this fold
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Fold {fold + 1}: Model saved to {model_save_path} with loss {best_loss:.4f}")

        scheduler.step()

    print(f"Fold {fold + 1}: Training completed. Best model saved to {model_save_path}")
    return model_save_path