import torch

data = torch.load('../pre_model/UniMol/Uni-Mol-3D.pt')
print(type(data))
if isinstance(data, dict):
    print(data.keys())