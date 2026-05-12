import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

def compute_label_stats(labels: np.ndarray):
    """
    计算每个标签的正负样本数、IRLbl、MeanIR、CVIR
    """
    n_samples, n_labels = labels.shape
    pos_counts = labels.sum(axis=0)
    neg_counts = n_samples - pos_counts
    maj_counts = np.maximum(pos_counts, neg_counts)
    min_counts = np.minimum(pos_counts, neg_counts)

    IRLbl = maj_counts / np.clip(min_counts, 1, None)
    MeanIR = IRLbl.mean()
    CVIR = IRLbl.std(ddof=1) / (MeanIR + 1e-12)

    return {
        "pos_counts": pos_counts,
        "neg_counts": neg_counts,
        "IRLbl": IRLbl,
        "MeanIR": MeanIR,
        "CVIR": CVIR
    }


def make_instance_weights(labels: np.ndarray, IRLbl: np.ndarray):
    """
    根据标签不平衡度为每个样本分配权重
    权重 = 该样本正标签对应 IRLbl 的平均
    """
    # 确保输入是numpy数组
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(IRLbl, torch.Tensor):
        IRLbl = IRLbl.cpu().numpy()

    n_samples, n_labels = labels.shape
    weights = []
    for i in range(n_samples):
        pos_idx = np.where(labels[i] == 1)[0]
        if len(pos_idx) > 0:
            wt = IRLbl[pos_idx].mean()
        else:
            wt = 0.5  # 纯负样本给个较低权重
        weights.append(wt)
    weights = np.array(weights, dtype=np.float64)
    return torch.DoubleTensor(weights)

def make_weighted_sampler(weights: torch.DoubleTensor):
    """
    根据样本权重构建 WeightedRandomSampler
    """
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler
