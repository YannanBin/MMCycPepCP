import torch
import numpy as np
from collections import namedtuple
from evaluate import compute_metrics

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, graph_data, morgan_fp, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            graph_data = graph_data.to(device)
            morgan_fp = morgan_fp.to(device)
            labels = labels.to(device)

            logits = model(input_ids, graph_data, morgan_fp, attention_mask=attention_mask)
            all_preds.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    Pred = namedtuple('Pred', ['predictions', 'label_ids'])
    pred = Pred(predictions=all_preds, label_ids=all_labels)
    metrics = compute_metrics(pred)
    return metrics