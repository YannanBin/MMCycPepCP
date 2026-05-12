import os
import pandas as pd

# 设置文件夹路径
pdb_folder = '../pdb_files'
output_csv = './pdb_filenames.csv'

# 获取所有.pdb文件名
pdb_files = [os.path.splitext(f)[0] for f in os.listdir(pdb_folder) if f.endswith('.pdb')]

# 创建DataFrame并保存为CSV
df = pd.DataFrame(pdb_files, columns=['PDB_Filename'])
df.to_csv(output_csv, index=False)

print(f"已保存{len(pdb_files)}个PDB文件名到 {output_csv}")
