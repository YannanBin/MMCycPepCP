import torch
import os
import numpy as np
# feature_dir = '../features/3d_pred_feature'
feature_dir = '../features/3d_feature'
features = []
for file in os.listdir(feature_dir):
    if file.endswith('.pt'):
        data = torch.load(os.path.join(feature_dir, file), weights_only=True)
        features.append(data['gnn_feature'].numpy())
features = np.vstack(features)
print(f"特征均值: {features.mean(axis=0)}")
print(f"特征方差: {features.var(axis=0)}")