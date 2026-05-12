from rdkit import Chem
import pandas as pd
from collections import Counter

data = pd.read_csv('../datasets/lastfinal_cyclic_peptides.csv')
atom_types = Counter()
for smiles in data['SMILES']:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"无效 SMILES: {smiles}")
        continue
    mol = Chem.AddHs(mol)  # 显式添加氢原子
    for atom in mol.GetAtoms():
        atom_types[atom.GetSymbol()] += 1
print("原子种类统计（含显式氢）:", dict(atom_types))