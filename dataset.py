import pandas as pd
import torch
from rdkit.Chem.rdDistGeom import EmbedMolecule
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils.data import Dataset
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from transformers import AutoTokenizer
from Bio.PDB import PDBParser, PPBuilder
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class CyclicPeptideDataset(Dataset):
    def __init__(self, csv_path, pre_model_path, max_len, pdb_dir, morgan_dir, graph_cache_dir):
        self.data = pd.read_csv(csv_path)
        self.max_len = max_len
        self.pdb_dir = pdb_dir
        self.morgan_dir = morgan_dir
        self.tokenizer = AutoTokenizer.from_pretrained(pre_model_path)
        # 添加图缓存
        self.graph_cache_dir = graph_cache_dir

        # 动态生成原子种类
        atom_types = set()
        for smiles in self.data['SMILES']:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mol = Chem.AddHs(mol)  # 显式添加氢
                for atom in mol.GetAtoms():
                    atom_types.add(atom.GetSymbol())
        self.atom_types = sorted(list(atom_types))
        print(f"检测到的原子种类: {self.atom_types}")

        self.label_cols = [col for col in self.data.columns if col not in ['CPKB ID', 'SMILES']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles = row['SMILES']
        cpkb_id = row['CPKB ID']
        labels = row[self.label_cols].values.astype(np.float32)

        # 编码 SMILES
        encoding = self.tokenizer(smiles, max_length=self.max_len, padding='max_length',
                                 truncation=True, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # Load or compute Morgan fingerprint
        if self.morgan_dir is not None:
            morgan_path = os.path.join(self.morgan_dir, f"{cpkb_id}.txt")
            if os.path.exists(morgan_path):
                try:
                    with open(morgan_path, 'r') as f:
                        morgan_fp = np.array([float(x) for x in f.read().strip().split()], dtype=np.float32)
                except Exception as e:
                    print(f"Error reading Morgan fingerprint {morgan_path}: {e}, computing from SMILES")
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        print(f"Invalid SMILES for {cpkb_id}: {smiles}")
                        morgan_fp = np.zeros(1024, dtype=np.float32)
                    else:
                        morgan_fp = np.array(GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024), dtype=np.float32)
            else:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    print(f"Invalid SMILES for {cpkb_id}: {smiles}")
                    morgan_fp = np.zeros(1024, dtype=np.float32)
                else:
                    morgan_fp = np.array(GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024), dtype=np.float32)
        else:
            # 如果没有 morgan_dir，就跳过或者使用全0向量
            morgan_fp = np.zeros(1024, dtype=np.float32)

        # Load or compute graph structure
        graph_cache_path = os.path.join(self.graph_cache_dir, f"{cpkb_id}.pt") if self.graph_cache_dir else None
        if graph_cache_path and os.path.exists(graph_cache_path):
            graph_data = torch.load(graph_cache_path)
        else:
            # Try loading PDB file
            pdb_path = os.path.join(self.pdb_dir, f"{cpkb_id}.pdb") if self.pdb_dir else None
            if pdb_path and os.path.exists(pdb_path):
                graph_data = self.pdb_to_graph(pdb_path, smiles)
            else:
                # Compute graph from SMILES (reference: pdbtoGraph.py)
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    print(f"Invalid SMILES for {cpkb_id}: {smiles}")
                    graph_data = Data(x=torch.zeros(10, len(self.atom_types) + 3),
                                      edge_index=torch.zeros(2, 0, dtype=torch.long))
                else:
                    graph_data = self.smiles_to_graph(mol)
            # Save to cache if graph_cache_dir is provided
            if graph_cache_path:
                os.makedirs(self.graph_cache_dir, exist_ok=True)
                torch.save(graph_data, graph_cache_path)

        morgan_fp = torch.from_numpy(morgan_fp).float()
        labels = torch.from_numpy(labels).float()

        return input_ids, attention_mask, graph_data, morgan_fp, labels

    def pdb_to_graph(self, pdb_path, smiles):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('peptide', pdb_path)
        atoms = []
        coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        atoms.append(atom)
                        coords.append(atom.get_coord())
        coords = np.array(coords, dtype=np.float32)
        coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-8)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        mol = Chem.AddHs(mol)  # 显式添加氢

        # 验证原子数量一致性
        if len(atoms) != mol.GetNumAtoms():
            print(f"警告: PDB {pdb_path} 原子数 {len(atoms)} 与 SMILES 原子数 {mol.GetNumAtoms()} 不一致")

        node_features = []
        for atom in atoms:
            element = atom.element
            atom_vec = [1 if element == at else 0 for at in self.atom_types]
            coord = atom.get_coord().flatten().tolist()
            node_features.append(atom_vec + coord)
        node_features = torch.tensor(node_features, dtype=torch.float)

        edge_index = []
        threshold = 2.25
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < threshold:
                    edge_index.append([i, j])
                    edge_index.append([j, i])

        if mol.GetNumConformers() > 0:
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                if [i, j] not in edge_index:
                    edge_index.append([i, j])
                    edge_index.append([j, i])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return Data(x=node_features, edge_index=edge_index)

    def smiles_to_graph(self, mol):
        if mol is None:
            return Data(x=torch.zeros(10, len(self.atom_types) + 3), edge_index=torch.zeros(2, 0, dtype=torch.long))

        mol = Chem.AddHs(mol)
        try:
            EmbedMolecule(mol, randomSeed=42, maxAttempts=50)
            MMFFOptimizeMolecule(mol, maxIters=200)
            conf = mol.GetConformer()
            coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())], dtype=np.float32)
            coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-8)
        except Exception as e:
            print(f"3D构象生成失败: {e}，使用零坐标")
            coords = np.zeros((mol.GetNumAtoms(), 3), dtype=np.float32)

        node_features = []
        for i, atom in enumerate(mol.GetAtoms()):
            element = atom.GetSymbol()
            atom_vec = [1 if element == at else 0 for at in self.atom_types]
            coord = coords[i].flatten().tolist()
            node_features.append(atom_vec + coord)
        node_features = torch.tensor(node_features, dtype=torch.float)

        edge_index = []
        threshold = 2.25
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < threshold:
                    edge_index.append([i, j])
                    edge_index.append([j, i])

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            if [i, j] not in edge_index:
                edge_index.append([i, j])
                edge_index.append([j, i])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        return Data(x=node_features, edge_index=edge_index)



# import pandas as pd
# import torch
# from rdkit.Chem.rdDistGeom import EmbedMolecule, ETKDGv3
# from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule, UFFOptimizeMolecule
# from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
# from torch.utils.data import Dataset
# import os
# import numpy as np
# from rdkit import Chem
# from rdkit.Chem import AllChem
# from torch_geometric.data import Data
# from transformers import AutoTokenizer
# from Bio.PDB import PDBParser, PPBuilder
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)
#
# class CyclicPeptideDataset(Dataset):
#     def __init__(self, csv_path, pre_model_path, max_len, pdb_dir, morgan_dir, graph_cache_dir, augment_prob=0.5, coord_noise_std=0.1):
#         self.data = pd.read_csv(csv_path)
#         self.max_len = max_len
#         self.pdb_dir = pdb_dir
#         self.morgan_dir = morgan_dir
#         self.graph_cache_dir = graph_cache_dir
#         self.tokenizer = AutoTokenizer.from_pretrained(pre_model_path)
#         self.augment_prob = augment_prob  # 增强概率
#         self.coord_noise_std = coord_noise_std  # 坐标扰动标准差
#
#         # 动态生成原子种类
#         atom_types = set()
#         for smiles in self.data['SMILES']:
#             mol = Chem.MolFromSmiles(smiles)
#             if mol:
#                 mol = Chem.AddHs(mol)  # 显式添加氢
#                 for atom in mol.GetAtoms():
#                     atom_types.add(atom.GetSymbol())
#         self.atom_types = sorted(list(atom_types))
#
#         self.label_cols = [col for col in self.data.columns if col not in ['CPKB ID', 'SMILES']]
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
#         smiles = row['SMILES']
#         cpkb_id = row['CPKB ID']
#         labels = row[self.label_cols].values.astype(np.float32)
#
#         # SMILES 增强
#         if np.random.random() < self.augment_prob:
#             mol = Chem.MolFromSmiles(smiles)
#             if mol:
#                 smiles = Chem.MolToSmiles(mol, doRandom=True)  # 随机化 SMILES
#                 print(f"增强 SMILES for {cpkb_id}: {smiles}")
#
#         # 编码 SMILES
#         encoding = self.tokenizer(smiles, max_length=self.max_len, padding='max_length',
#                                  truncation=True, return_tensors='pt')
#         input_ids = encoding['input_ids'].squeeze()
#         attention_mask = encoding['attention_mask'].squeeze()
#
#         # Load or compute Morgan fingerprint
#         if self.morgan_dir is not None:
#             morgan_path = os.path.join(self.morgan_dir, f"{cpkb_id}.txt")
#             if os.path.exists(morgan_path):
#                 try:
#                     with open(morgan_path, 'r') as f:
#                         morgan_fp = np.array([float(x) for x in f.read().strip().split()], dtype=np.float32)
#                 except Exception:
#                     mol = Chem.MolFromSmiles(smiles)
#                     if mol is None:
#                         morgan_fp = np.zeros(1024, dtype=np.float32)
#                     else:
#                         morgan_fp = np.array(GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024), dtype=np.float32)
#             else:
#                 mol = Chem.MolFromSmiles(smiles)
#                 if mol is None:
#                     morgan_fp = np.zeros(1024, dtype=np.float32)
#                 else:
#                     morgan_fp = np.array(GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024), dtype=np.float32)
#         else:
#             morgan_fp = np.zeros(1024, dtype=np.float32)
#
#         # Load or compute graph structure
#         graph_cache_path = os.path.join(self.graph_cache_dir, f"{cpkb_id}.pt") if self.graph_cache_dir else None
#         if graph_cache_path and os.path.exists(graph_cache_path) and np.random.random() >= self.augment_prob:
#             graph_data = torch.load(graph_cache_path)
#         else:
#             pdb_path = os.path.join(self.pdb_dir, f"{cpkb_id}.pdb") if self.pdb_dir else None
#             if pdb_path and os.path.exists(pdb_path):
#                 graph_data = self.pdb_to_graph(pdb_path, smiles, augment=np.random.random() < self.augment_prob)
#             else:
#                 mol = Chem.MolFromSmiles(smiles)
#                 if mol is None:
#                     graph_data = Data(x=torch.zeros(10, len(self.atom_types) + 3),
#                                       edge_index=torch.zeros(2, 0, dtype=torch.long))
#                 else:
#                     graph_data = self.smiles_to_graph(mol, augment=np.random.random() < self.augment_prob)
#             if graph_cache_path and np.random.random() >= self.augment_prob:
#                 os.makedirs(self.graph_cache_dir, exist_ok=True)
#                 torch.save(graph_data, graph_cache_path)
#
#         morgan_fp = torch.from_numpy(morgan_fp).float()
#         labels = torch.from_numpy(labels).float()
#
#         return input_ids, attention_mask, graph_data, morgan_fp, labels
#
#     def predict_3d_conformation(self, mol, augment=False):
#         """Generate 3D conformation with ETKDGv3 embedding and MMFF/UFF optimization."""
#         try:
#             # Add hydrogens and embed with ETKDGv3
#             mol_3d = Chem.AddHs(mol.__copy__())
#             etkdg = ETKDGv3()
#             etkdg.randomSeed = np.random.randint(1, 1000) if augment else 42
#             etkdg.maxIterations = 200
#             etkdg.useRandomCoords = True
#             etkdg.numThreads = 1  # Avoid nested parallelism
#
#             # Generate single conformation
#             if EmbedMolecule(mol_3d, etkdg) != 0:
#                 return None
#
#             # 优先使用 MMFF 优化
#             try:
#                 MMFFOptimizeMolecule(mol_3d, maxIters=200)
#             except Exception:
#                 # MMFF 失败时回退到 UFF
#                 try:
#                     UFFOptimizeMolecule(mol_3d, maxIters=200)
#                 except Exception:
#                     return None
#
#             return mol_3d if mol_3d.GetNumConformers() > 0 else None
#         except Exception:
#             return None
#
#     def pdb_to_graph(self, pdb_path, smiles, augment=False):
#         parser = PDBParser(QUIET=True)
#         structure = parser.get_structure('peptide', pdb_path)
#         atoms = []
#         coords = []
#         for model in structure:
#             for chain in model:
#                 for residue in chain:
#                     for atom in residue:
#                         atoms.append(atom)
#                         coords.append(atom.get_coord())
#         coords = np.array(coords, dtype=np.float32)
#
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is None:
#             raise ValueError(f"Invalid SMILES: {smiles}")
#         mol = Chem.AddHs(mol)
#
#         if len(atoms) != mol.GetNumAtoms():
#             print(f"警告: PDB {pdb_path} 原子数 {len(atoms)} 与 SMILES 原子数 {mol.GetNumAtoms()} 不一致")
#
#         # 3D 结构增强
#         if augment:
#             if np.random.random() < 0.5:  # 50% 概率进行坐标扰动
#                 noise = np.random.normal(0, self.coord_noise_std, coords.shape)
#                 coords = coords + noise
#                 print(f"增强坐标 for {pdb_path}: 坐标扰动")
#             else:  # 50% 概率生成新构象
#                 mol_3d = self.predict_3d_conformation(mol, augment=True)
#                 if mol_3d is None:
#                     print(f"3D构象生成失败 for {pdb_path}, 使用原始坐标")
#                 else:
#                     conf = mol_3d.GetConformer()
#                     coords = np.array([conf.GetAtomPosition(i) for i in range(mol_3d.GetNumAtoms())], dtype=np.float32)
#                     print(f"增强坐标 for {pdb_path}: 生成新构象")
#
#         coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-8)
#
#         node_features = []
#         for i, atom in enumerate(atoms):
#             element = atom.element
#             atom_vec = [1 if element == at else 0 for at in self.atom_types]
#             coord = coords[i].flatten().tolist()
#             node_features.append(atom_vec + coord)
#         node_features = torch.tensor(node_features, dtype=torch.float)
#
#         edge_index = []
#         threshold = 2.25
#         for i in range(len(atoms)):
#             for j in range(i + 1, len(atoms)):
#                 dist = np.linalg.norm(coords[i] - coords[j])
#                 if dist < threshold:
#                     edge_index.append([i, j])
#                     edge_index.append([j, i])
#
#         if mol.GetNumConformers() > 0:
#             for bond in mol.GetBonds():
#                 i = bond.GetBeginAtomIdx()
#                 j = bond.GetEndAtomIdx()
#                 if [i, j] not in edge_index:
#                     edge_index.append([i, j])
#                     edge_index.append([j, i])
#
#         edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#         return Data(x=node_features, edge_index=edge_index)
#
#     def smiles_to_graph(self, mol, augment=False):
#         if mol is None:
#             return Data(x=torch.zeros(10, len(self.atom_types) + 3), edge_index=torch.zeros(2, 0, dtype=torch.long))
#
#         mol = Chem.AddHs(mol)
#         mol_3d = self.predict_3d_conformation(mol, augment=augment)
#         if mol_3d is None:
#             coords = np.zeros((mol.GetNumAtoms(), 3), dtype=np.float32)
#         else:
#             conf = mol_3d.GetConformer()
#             coords = np.array([conf.GetAtomPosition(i) for i in range(mol_3d.GetNumAtoms())], dtype=np.float32)
#
#         # 3D 结构增强：坐标扰动
#         if augment and np.random.random() < 0.5 and mol_3d is not None:
#             noise = np.random.normal(0, self.coord_noise_std, coords.shape)
#             coords = coords + noise
#             print(f"增强坐标 for SMILES: 坐标扰动")
#
#         coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-8)
#
#         node_features = []
#         for i, atom in enumerate(mol.GetAtoms()):
#             element = atom.GetSymbol()
#             atom_vec = [1 if element == at else 0 for at in self.atom_types]
#             coord = coords[i].flatten().tolist()
#             node_features.append(atom_vec + coord)
#         node_features = torch.tensor(node_features, dtype=torch.float)
#
#         edge_index = []
#         threshold = 2.25
#         for i in range(len(coords)):
#             for j in range(i + 1, len(coords)):
#                 dist = np.linalg.norm(coords[i] - coords[j])
#                 if dist < threshold:
#                     edge_index.append([i, j])
#                     edge_index.append([j, i])
#
#         for bond in mol.GetBonds():
#             i = bond.GetBeginAtomIdx()
#             j = bond.GetEndAtomIdx()
#             if [i, j] not in edge_index:
#                 edge_index.append([i, j])
#                 edge_index.append([j, i])
#
#         edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#
#         return Data(x=node_features, edge_index=edge_index)
