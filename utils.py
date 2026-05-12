import yaml
import random
import numpy as np
import torch
from torch_geometric.data import Batch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def custom_collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])
    attention_mask = torch.stack([item[1] for item in batch])
    graph_data = Batch.from_data_list([item[2] for item in batch])
    morgan_fp = torch.stack([item[3] for item in batch])
    labels = torch.stack([item[4] for item in batch])
    return input_ids, attention_mask, graph_data, morgan_fp, labels