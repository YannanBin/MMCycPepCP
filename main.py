import torch
import argparse
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from dataset import CyclicPeptideDataset
from model import MultiModalPeptideModel
from test import evaluate_model
from train import train_model
from torch.utils.data import DataLoader, Subset
from utils import set_seed, load_config, custom_collate_fn
from torch_geometric.data import Batch



def main():
    parser = argparse.ArgumentParser(description="多模态环肽多标签预测（五折交叉验证）")
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config(args.config)

    dataset = CyclicPeptideDataset(config['data']['data_path'], config['model']['pre_model_path'],
                                   config['data']['max_len'], config['data']['pdb_dir'],
                                   config['data']['morgan_dir'], config['data']['graph_cache_dir'])
    # dataset = CyclicPeptideDataset(config['data']['data_path'], config['model']['pre_model_path'],
    #                                config['data']['max_len'], config['data']['pdb_dir'],
    #                                config['data']['morgan_dir'], config['data']['graph_cache_dir'],
    #                                config['data']['augment_prob'], config['data']['coord_noise_std'])

    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    all_metrics = []

    model_save_paths = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(len(dataset)))):
        print(f"\nFold {fold + 1}/5")

        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        train_loader = DataLoader(train_subset, batch_size=config['train']['batch_size'],
                                 shuffle=True, collate_fn=custom_collate_fn,
                                 num_workers=config['train']['num_workers'])
        test_loader = DataLoader(test_subset, batch_size=config['train']['batch_size'],
                                shuffle=False, collate_fn=custom_collate_fn,
                                num_workers=config['train']['num_workers'])

        model = MultiModalPeptideModel(dataset.atom_types,
                                       pre_model_path=config['model']['pre_model_path'],
                                       transformer_layer=config['model']['transformer_layer'],
                                       num_labels=config['model']['num_labels'],
                                       gnn_hidden_dim=config['model']['gnn_hidden_dim'],
                                       cnn_hidden_dim=config['model']['cnn_hidden_dim'],
                                       tcf_layer=config['model']['tcf_layer'],
                                       tcf_interval=config['model']['tcf_interval'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # train_model(model, train_loader, config, device)

        # 修改
        # train_model(model, train_loader, config, device, dataset)

        model_save_path = train_model(model, train_loader, config, device, dataset, fold)
        model_save_paths.append(model_save_path)


        metrics = evaluate_model(model, test_loader, device)
        all_metrics.append(metrics)
        print(f"Fold {fold + 1} Metrics: {metrics}")

    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    print("\nFive-Fold Average Metrics:")
    for key, value in avg_metrics.items():
        print(f"{key}: {value:.4f}")

    # Save model paths to a file for use in visualization
    with open('model_paths.txt', 'w') as f:
        for path in model_save_paths:
            f.write(f"{path}\n")
    print("Model paths saved to 'model_paths.txt'")

if __name__ == "__main__":
    main()