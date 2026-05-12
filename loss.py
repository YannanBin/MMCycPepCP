import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================= 工具函数 =================

def compute_label_weights(labels, max_weight=5.0):
    """
    计算每个标签的正样本权重 (pos_weight)
    用于 BCE / Focal / ASL
    """
    n_samples, n_labels = labels.shape
    pos_weights = []
    for i in range(n_labels):
        pos_count = np.sum(labels[:, i])
        neg_count = n_samples - pos_count
        if pos_count == 0:
            weight = 1.0
        else:
            weight = min(neg_count / pos_count, max_weight)
        pos_weights.append(weight)
    print(f"[Loss] 正样本权重: {pos_weights}")
    return torch.tensor(pos_weights, dtype=torch.float)


def class_balanced_weights(pos_counts, beta=0.9999):
    """
    计算每个类别的 Class-Balanced 权重 (Cui et al. 2019)
    pos_counts: ndarray [num_labels] 每个标签正样本数
    """
    effective_num = 1.0 - np.power(beta, pos_counts)
    weights = (1.0 - beta) / (effective_num + 1e-8)
    weights = weights / np.sum(weights) * len(pos_counts)  # normalize
    return torch.tensor(weights, dtype=torch.float)


# ================== 各类 Loss ==================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weights=None, reduction='mean'):
        """
        alpha: float | Tensor[num_labels] | 'auto'
            - float: 固定权重
            - Tensor: 每个类别独立 alpha
            - 'auto': 根据标签频率自动计算 alpha
        gamma: 聚焦因子
        pos_weights: Tensor[num_labels], 正样本权重 (来自 neg/pos 比例)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weights = pos_weights
        self.reduction = reduction
        self.alpha_vector = None  # 存储 per-class alpha

    def compute_alpha_auto(self, labels):
        """ 根据标签统计自动生成 alpha 向量 """
        n_samples, n_labels = labels.shape
        alphas = []
        for i in range(n_labels):
            pos = labels[:, i].sum()
            neg = n_samples - pos
            if pos == 0:
                alpha_c = 0.5
            else:
                alpha_c = neg / (pos + neg)
            alpha_c = min(0.95, max(0.05, alpha_c))  # 截断
            alphas.append(alpha_c)
        return torch.tensor(alphas, dtype=torch.float)

    def forward(self, logits, labels):
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        pt = labels * probs + (1 - labels) * (1 - probs)

        # --- alpha 支持多种形式 ---
        if isinstance(self.alpha, torch.Tensor):
            alpha = self.alpha.to(logits.device).view(1, -1)
        elif isinstance(self.alpha, float):
            alpha = self.alpha
        else:
            raise ValueError("alpha 必须是 float 或 Tensor")

        focal_weight = alpha * (1 - pt).clamp(min=1e-8) ** self.gamma

        if self.pos_weights is not None:
            pos_weights = self.pos_weights.to(logits.device).view(1, -1)
            weight_mask = labels * pos_weights + (1 - labels)
            focal_weight = focal_weight * weight_mask

        loss = focal_weight * ce_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()


class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, beta=0.9999, gamma=2.0, alpha=0.25,
                 samples_per_class=None, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.samples_per_class = samples_per_class
        self.reduction = reduction

    def forward(self, logits, labels):
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')

        if self.samples_per_class is not None:
            samples = self.samples_per_class.to(logits.device)
            effective_num = 1.0 - torch.pow(self.beta, samples)
            weights = (1.0 - self.beta) / (effective_num + 1e-8)
            weights = weights / weights.sum() * len(samples)
            weights = weights.view(1, -1)
        else:
            weights = torch.ones(1, logits.size(1), device=logits.device)

        pt = torch.where(labels == 1, probs, 1 - probs)
        focal_factor = self.alpha * (1 - pt).clamp(min=1e-8) ** self.gamma
        loss = focal_factor * ce_loss * weights
        return loss.mean() if self.reduction == 'mean' else loss.sum()


class ClassBalancedCELoss(nn.Module):
    """ 纯 Class-Balanced CrossEntropy，不带 Focal """
    def __init__(self, beta=0.9999, samples_per_class=None, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.samples_per_class = samples_per_class
        self.reduction = reduction

    def forward(self, logits, labels):
        ce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        if self.samples_per_class is not None:
            samples = self.samples_per_class.to(logits.device)
            effective_num = 1.0 - torch.pow(self.beta, samples)
            weights = (1.0 - self.beta) / (effective_num + 1e-8)
            weights = weights / weights.sum() * len(samples)
            weights = weights.view(1, -1)
        else:
            weights = torch.ones(1, logits.size(1), device=logits.device)
        loss = ce_loss * weights
        return loss.mean() if self.reduction == 'mean' else loss.sum()


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=1.0, gamma_neg=4.0,
                 pos_weights=None, reduction='mean'):
        super().__init__()
        # gamma_pos / gamma_neg 可以是 float 或 Tensor
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.pos_weights = pos_weights
        self.reduction = reduction

    def forward(self, logits, labels):
        probs = torch.sigmoid(logits)
        log_probs = torch.log(probs.clamp(min=1e-8))
        log_one_minus = torch.log((1 - probs).clamp(min=1e-8))

        # --- gamma 支持 per-class ---
        if isinstance(self.gamma_pos, torch.Tensor):
            gamma_pos = self.gamma_pos.to(logits.device).view(1, -1)
        else:
            gamma_pos = self.gamma_pos

        if isinstance(self.gamma_neg, torch.Tensor):
            gamma_neg = self.gamma_neg.to(logits.device).view(1, -1)
        else:
            gamma_neg = self.gamma_neg

        pos_loss = -labels * (1 - probs) ** gamma_pos * log_probs
        neg_loss = -(1 - labels) * probs ** gamma_neg * log_one_minus

        # --- pos_weight 保留 ---
        if self.pos_weights is not None:
            pos_weights = self.pos_weights.to(logits.device).view(1, -1)
            weight_mask = labels * pos_weights + (1 - labels)
            pos_loss = pos_loss * weight_mask
            neg_loss = neg_loss * weight_mask

        loss = pos_loss + neg_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()



class LogitAdjustedLoss(nn.Module):
    """
    Logit-Adjusted BCE (Menon et al. 2020)
    logits -> logits + log(p/(1-p)) before BCE
    """
    def __init__(self, priors, tau=1.0, reduction='mean'):
        super().__init__()
        self.priors = priors  # Tensor[num_labels], 每个标签正样本概率
        self.tau = tau
        self.reduction = reduction

    def forward(self, logits, labels):
        bias = torch.log(self.priors / (1 - self.priors)).to(logits.device)
        adjusted_logits = logits + self.tau * bias
        loss = F.binary_cross_entropy_with_logits(adjusted_logits, labels, reduction='none')
        return loss.mean() if self.reduction == 'mean' else loss.sum()


class LabelSmoothingLoss(nn.Module):
    """ 多标签版 Label Smoothing """
    def __init__(self, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits, labels):
        smooth_labels = labels * (1 - self.smoothing) + 0.5 * self.smoothing
        loss = F.binary_cross_entropy_with_logits(logits, smooth_labels, reduction='none')
        return loss.mean() if self.reduction == 'mean' else loss.sum()


# ============== Loss 工厂方法 ==============

def get_loss(config, labels=None):
    """
    根据 config['train']['loss_type'] 返回相应的损失函数
    """
    loss_type = config['train']['loss_type'].lower()

    # if loss_type == 'focal':
    #     pos_weights = compute_label_weights(labels) if labels is not None else None
    #     return FocalLoss(alpha=config['train'].get('focal_alpha', 0.25),
    #                      gamma=config['train'].get('focal_gamma', 2.0),
    #                      pos_weights=pos_weights)
    if loss_type == 'focal':
        pos_weights = compute_label_weights(labels) if labels is not None else None

        # 新增：alpha 自动计算
        alpha_cfg = config['train'].get('focal_alpha', 0.25)
        if alpha_cfg == 'auto' and labels is not None:
            n_samples, n_labels = labels.shape
            alphas = []
            for i in range(n_labels):
                pos = labels[:, i].sum()
                neg = n_samples - pos
                alpha_c = neg / (pos + neg) if pos > 0 else 0.5
                alpha_c = min(0.95, max(0.05, alpha_c))
                alphas.append(alpha_c)
            alpha = torch.tensor(alphas, dtype=torch.float)
            print(f"[Loss] 自动计算 per-class alpha: {alpha}")
        else:
            alpha = alpha_cfg  # float

        return FocalLoss(alpha=alpha,
                         gamma=config['train'].get('focal_gamma', 2.0),
                         pos_weights=pos_weights)

    elif loss_type == 'cb focal':
        samples = torch.tensor(labels.sum(axis=0), dtype=torch.float) if labels is not None else None
        return ClassBalancedFocalLoss(beta=config['train'].get('cb_beta', 0.9999),
                                      gamma=config['train'].get('cb_gamma', 2.0),
                                      alpha=config['train'].get('cb_alpha', 0.25),
                                      samples_per_class=samples)

    elif loss_type == 'cb ce':
        samples = torch.tensor(labels.sum(axis=0), dtype=torch.float) if labels is not None else None
        return ClassBalancedCELoss(beta=config['train'].get('cb_beta', 0.9999),
                                   samples_per_class=samples)

    # elif loss_type == 'asl':
    #     pos_weights = compute_label_weights(labels) if labels is not None else None
    #     return AsymmetricLoss(gamma_pos=config['train'].get('asl_gamma_pos', 1.0),
    #                           gamma_neg=config['train'].get('asl_gamma_neg', 4.0),
    #                           pos_weights=pos_weights)

    elif loss_type == 'asl':
        pos_weights = compute_label_weights(labels) if labels is not None else None

        gamma_pos = config['train'].get('asl_gamma_pos', 1.0)
        gamma_neg = config['train'].get('asl_gamma_neg', 4.0)

        # === 新增：自动 gamma_neg 计算 ===
        if gamma_neg == "auto" and labels is not None:
            n_samples, n_labels = labels.shape
            ir = []
            for i in range(n_labels):
                pos = labels[:, i].sum()
                neg = n_samples - pos
                ir.append(neg / (pos + 1e-8))
            ir = np.array(ir)

            # 线性映射到 [2,5]
            gamma_neg_values = 2.0 + 3.0 * (ir / ir.max())
            gamma_neg = torch.tensor(gamma_neg_values, dtype=torch.float)
            print(f"[Loss] 自动计算 per-class gamma_neg: {gamma_neg}")

        return AsymmetricLoss(
            gamma_pos=gamma_pos,
            gamma_neg=gamma_neg,
            pos_weights=pos_weights
        )

    elif loss_type == 'logit-adjust':
        priors = torch.tensor(labels.mean(axis=0), dtype=torch.float) if labels is not None else None
        return LogitAdjustedLoss(priors, tau=config['train'].get('tau', 1.0))

    elif loss_type == 'label-smoothing':
        return LabelSmoothingLoss(smoothing=config['train'].get('smoothing', 0.1))

    else:
        pos_weights = compute_label_weights(labels) if labels is not None else None
        # return nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        return nn.BCEWithLogitsLoss(pos_weight=None)
