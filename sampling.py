import torch
from torch.utils.data import DataLoader, Subset
from utils import custom_collate_fn


def select_hard_negatives(model, dataset, config, k=5, confidence_threshold=0.5):
    """
    Select hard negative samples based on model predictions for multi-modal peptide dataset.

    Args:
        model (nn.Module): Current trained MultiModalPeptideModel
        dataset (CyclicPeptideDataset): Dataset containing samples for negative mining
        config (dict): Configuration dictionary containing training parameters
        k (int): Number of hard negative samples to select per class
        confidence_threshold (float): Threshold for selecting hard negatives

    Returns:
        torch.Tensor: Indices of selected hard negative samples
    """
    model.eval()
    device = next(model.parameters()).device
    batch_size = config['train']['batch_size']

    # Create DataLoader for the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=custom_collate_fn, num_workers=config['train']['num_workers'])

    all_scores = []
    all_indices = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask, graph_data, morgan_fp, labels) in enumerate(data_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            graph_data = graph_data.to(device)
            morgan_fp = morgan_fp.to(device)
            labels = labels.to(device)

            # Get model predictions
            logits = model(input_ids, graph_data, morgan_fp, attention_mask=attention_mask)
            probs = torch.sigmoid(logits)

            # Find negative samples (where label is 0)
            negative_mask = (labels == 0)

            # Calculate scores for negative samples
            # Higher score means the model wrongly predicted positive with high confidence
            wrong_positive_probs = probs * negative_mask

            # Get maximum prediction score across all classes for each sample
            sample_scores = wrong_positive_probs.max(dim=1)[0]

            # Only keep samples above the confidence threshold
            valid_samples = sample_scores > confidence_threshold
            if valid_samples.any():
                batch_indices = torch.arange(input_ids.size(0), device=device) + batch_idx * batch_size
                all_scores.append(sample_scores[valid_samples].cpu())
                all_indices.append(batch_indices[valid_samples].cpu())
                all_labels.append(labels[valid_samples].cpu())

    if not all_scores:
        print("No hard negative samples found above confidence threshold.")
        return torch.tensor([], dtype=torch.long)

    # Combine all batches
    all_scores = torch.cat(all_scores)
    all_indices = torch.cat(all_indices)
    all_labels = torch.cat(all_labels)

    # Select hard negatives for each class
    selected_indices = []
    n_classes = all_labels.size(1)

    for class_idx in range(n_classes):
        # Get negative samples for current class
        class_negative_mask = (all_labels[:, class_idx] == 0)
        class_scores = all_scores[class_negative_mask]
        class_indices = all_indices[class_negative_mask]

        if len(class_scores) > 0:
            # Select top k samples
            k_actual = min(k, len(class_scores))
            _, hard_neg_idx = torch.topk(class_scores, k_actual)
            selected_indices.append(class_indices[hard_neg_idx])

    # Combine and deduplicate selected indices
    if selected_indices:
        selected_indices = torch.cat(selected_indices)
        selected_indices = torch.unique(selected_indices)
        print(f"Selected {len(selected_indices)} hard negative samples.")
        return selected_indices
    else:
        print("No hard negative samples selected.")
        return torch.tensor([], dtype=torch.long)


def select_hard_positives(model, dataset, config, k=5, confidence_threshold=0.5):
    """
    Select hard positive samples based on low confidence predictions for positive labels.

    Args:
        model (nn.Module): Trained model.
        dataset (Dataset): Dataset to evaluate.
        config (dict): Config dict.
        k (int): Number of hard positives per class.
        confidence_threshold (float): Max confidence to consider a positive sample as hard.

    Returns:
        torch.Tensor: Indices of selected hard positive samples.
    """
    model.eval()
    device = next(model.parameters()).device
    batch_size = config['train']['batch_size']

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=custom_collate_fn, num_workers=config['train']['num_workers'])

    all_scores = []
    all_indices = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask, graph_data, morgan_fp, labels) in enumerate(data_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            graph_data = graph_data.to(device)
            morgan_fp = morgan_fp.to(device)
            labels = labels.to(device)

            logits = model(input_ids, graph_data, morgan_fp, attention_mask=attention_mask)
            probs = torch.sigmoid(logits)

            positive_mask = (labels == 1)
            low_confidence_probs = probs * positive_mask  # 保留正样本对应的预测概率

            # 最低预测概率作为困难程度
            sample_scores = low_confidence_probs.min(dim=1)[0]

            # 筛选出低于阈值的正样本
            valid_samples = sample_scores < confidence_threshold
            if valid_samples.any():
                batch_indices = torch.arange(input_ids.size(0), device=device) + batch_idx * batch_size
                all_scores.append(sample_scores[valid_samples].cpu())
                all_indices.append(batch_indices[valid_samples].cpu())
                all_labels.append(labels[valid_samples].cpu())

    if not all_scores:
        print("No hard positive samples found below confidence threshold.")
        return torch.tensor([], dtype=torch.long)

    all_scores = torch.cat(all_scores)
    all_indices = torch.cat(all_indices)
    all_labels = torch.cat(all_labels)

    selected_indices = []
    n_classes = all_labels.size(1)

    for class_idx in range(n_classes):
        class_positive_mask = (all_labels[:, class_idx] == 1)
        class_scores = all_scores[class_positive_mask]
        class_indices = all_indices[class_positive_mask]

        if len(class_scores) > 0:
            k_actual = min(k, len(class_scores))
            _, hard_pos_idx = torch.topk(-class_scores, k_actual)  # 越小越“困难”
            selected_indices.append(class_indices[hard_pos_idx])

    if selected_indices:
        selected_indices = torch.cat(selected_indices)
        selected_indices = torch.unique(selected_indices)
        print(f"Selected {len(selected_indices)} hard positive samples.")
        return selected_indices
    else:
        print("No hard positive samples selected.")
        return torch.tensor([], dtype=torch.long)
