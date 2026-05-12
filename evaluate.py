import numpy as np
from sklearn.metrics import precision_score, recall_score


def Aiming(y_hat, y):
    '''
    The “Aiming” rate (also called “Precision”) is to reflect the average ratio of the
    correctly predicted labels over the predicted labels; to measure the percentage
    of the predicted labels that hit the target of the real labels.
    '''
    n, m = y_hat.shape
    sorce_k = 0
    valid_samples = 0
    for v in range(n):
        intersection = np.sum((y_hat[v] == 1) & (y[v] == 1))
        predicted_positives = np.sum(y_hat[v])
        if predicted_positives == 0:
            sorce_k += 0  # 无预测正标签，贡献 0 分数
        else:
            sorce_k += intersection / predicted_positives
        valid_samples += 1
    return sorce_k / valid_samples if valid_samples > 0 else 0.0


def Coverage(y_hat, y):
    '''
    The “Coverage” rate (also called “Recall”) is to reflect the average ratio of the
    correctly predicted labels over the real labels; to measure the percentage of the
    real labels that are covered by the hits of prediction.
    '''
    n, m = y_hat.shape
    sorce_k = 0
    valid_samples = 0
    for v in range(n):
        intersection = np.sum((y_hat[v] == 1) & (y[v] == 1))
        true_positives = np.sum(y[v])
        if true_positives == 0:
            sorce_k += 0  # 无真实正标签，贡献 0 分数
        else:
            sorce_k += intersection / true_positives
        valid_samples += 1
    return sorce_k / valid_samples if valid_samples > 0 else 0.0


def Accuracy(y_hat, y):
    '''
    The “Accuracy” rate is to reflect the average ratio of correctly predicted labels
    over the total labels including correctly and incorrectly predicted labels as well
    as those real labels but are missed in the prediction.
    '''
    n, m = y_hat.shape
    sorce_k = 0
    valid_samples = 0
    for v in range(n):
        intersection = np.sum((y_hat[v] == 1) & (y[v] == 1))
        union = np.sum((y_hat[v] == 1) | (y[v] == 1))
        if union == 0:
            sorce_k += 0  # 无预测或真实正标签，贡献 0 分数
        else:
            sorce_k += intersection / union
        valid_samples += 1
    return sorce_k / valid_samples if valid_samples > 0 else 0.0


def AbsoluteTrue(y_hat, y):
    '''
    The proportion of samples where the predicted label set exactly matches the true label set.
    '''
    n, m = y_hat.shape
    sorce_k = 0
    for v in range(n):
        if list(y_hat[v]) == list(y[v]):
            sorce_k += 1
    return sorce_k / n


def AbsoluteFalse(y_hat, y):
    '''
    The “AbsoluteFalse” rate (also called “Hamming Loss”) reflects the average ratio of
    incorrect labels (predicted but not true, or true but not predicted) over the total labels.
    '''
    n, m = y_hat.shape
    sorce_k = 0
    for v in range(n):
        union = np.sum((y_hat[v] == 1) | (y[v] == 1))
        intersection = np.sum((y_hat[v] == 1) & (y[v] == 1))
        sorce_k += (union - intersection) / m
    return sorce_k / n


def compute_metrics(pred):
    """
    Compute multi-label classification metrics:
    - Aiming (Precision): Ratio of correctly predicted labels over predicted labels.
    - Coverage (Recall): Ratio of correctly predicted labels over true labels.
    - Accuracy: Ratio of correctly predicted labels over union of predicted and true labels.
    - AbsoluteTrue: Proportion of samples with exact label set match.
    - AbsoluteFalse: Hamming Loss, proportion of incorrect labels.
    - Precision (micro): Global precision across all labels.
    - Recall (micro): Global recall across all labels.
    """
    # Convert logits to binary predictions (threshold 0.5 to encourage positive predictions)
    predictions = (pred.predictions > 0.5).astype(int)
    # Convert float labels to int
    labels = pred.label_ids.astype(int)

    # 统计预测分布
    # predicted_positives = np.mean(np.sum(predictions, axis=1))
    # true_positives = np.mean(np.sum(labels, axis=1))
    # print(f"预测平均正标签数: {predicted_positives:.2f}, 真实平均正标签数: {true_positives:.2f}")
    # 每个标签的正预测比例
    # pred_pos_ratios = np.mean(predictions, axis=0)
    # true_pos_ratios = np.mean(labels, axis=0)
    # print("每个标签的预测正比例 vs 真实正比例:")
    # for i, (pred_ratio, true_ratio) in enumerate(zip(pred_pos_ratios, true_pos_ratios)):
    #     print(f"  标签 {i}: 预测 {pred_ratio:.4f}, 真实 {true_ratio:.4f}")

    metrics = {
        'precision': Aiming(predictions, labels),
        'coverage': Coverage(predictions, labels),
        'accuracy': Accuracy(predictions, labels),
        'absolute_true': AbsoluteTrue(predictions, labels),
        'absolute_false': AbsoluteFalse(predictions, labels),
        'precision_micro': precision_score(labels, predictions, average='micro', zero_division=0),
        'recall_micro': recall_score(labels, predictions, average='micro', zero_division=0)
    }

    return metrics