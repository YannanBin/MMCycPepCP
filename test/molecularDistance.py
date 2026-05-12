import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser
import argparse
from utils import load_config
from dataset import CyclicPeptideDataset
from scipy import stats

def analyze_distance_distribution(csv_path, pdb_dir, max_len, output_dir):
    """
    分析 PDB 文件中原子间距离的分布，生成直方图和统计信息。

    参数：
        csv_path (str): 数据集 CSV 文件路径。
        pdb_dir (str): 包含 PDB 文件的目录。
        max_len (int): 最大序列长度（保持兼容性）。
        output_dir (str): 保存直方图和统计信息的输出目录。
    """
    # 初始化数据集以访问 pdb_to_graph 方法
    # dataset = CyclicPeptideDataset(csv_path, pre_model_path=None, max_len=max_len, pdb_dir=pdb_dir,
    #                                morgan_feature_dir=None, pdb_feature_dir=None, morgan_dir=None)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取数据集 CSV 文件
    data = pd.read_csv(csv_path)
    total_files = len(data)
    distances = []
    processed = 0
    skipped = 0

    print(f"正在处理 {total_files} 个 PDB 文件以分析距离分布...")

    # 遍历每个 PDB 文件
    for idx, row in data.iterrows():
        cpkb_id = row['CPKB ID']
        smiles = row['SMILES']
        pdb_path = os.path.join(pdb_dir, f"{cpkb_id}.pdb")

        try:
            # 检查 PDB 文件是否存在
            if not os.path.exists(pdb_path):
                print(f"未找到 {cpkb_id} 的 PDB 文件 {pdb_path}，跳过")
                skipped += 1
                continue

            # 解析 PDB 文件
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('peptide', pdb_path)
            coords = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            coords.append(atom.get_coord())
            coords = np.array(coords, dtype=np.float32)

            # 标准化坐标（与 pdb_to_graph 一致）
            coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-8)

            # 计算所有原子对之间的距离
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    dist = np.linalg.norm(coords[i] - coords[j])
                    distances.append(dist)

            processed += 1
            print(f"已处理 {cpkb_id} 的距离数据")

        except Exception as e:
            print(f"处理 {cpkb_id} 时出错：{e}，跳过")
            skipped += 1
            continue

    # 转换为 numpy 数组
    distances = np.array(distances)

    # 计算统计信息
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    median_dist = np.median(distances)
    percentiles = np.percentile(distances, [25, 50, 75, 90, 95])

    # 打印统计信息
    print(f"\n距离分布统计：")
    print(f"样本数：{len(distances)}")
    print(f"均值：{mean_dist:.3f}")
    print(f"标准差：{std_dist:.3f}")
    print(f"中位数：{median_dist:.3f}")
    print(f"25% 分位数：{percentiles[0]:.3f}")
    print(f"50% 分位数：{percentiles[1]:.3f}")
    print(f"75% 分位数：{percentiles[2]:.3f}")
    print(f"90% 分位数：{percentiles[3]:.3f}")
    print(f"95% 分位数：{percentiles[4]:.3f}")

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=100, density=True, alpha=0.7, color='blue', label='距离分布')
    plt.axvline(x=4.0, color='red', linestyle='--', label='当前阈值 (4.0)')
    plt.xlabel('标准化距离')
    plt.ylabel('密度')
    plt.title('原子间标准化距离分布')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 保存直方图
    # hist_path = os.path.join(output_dir, 'distance_histogram.png')
    # plt.savefig(hist_path)
    # plt.close()

    # print(f"直方图已保存到 {hist_path}")

    # 提供阈值建议
    print("\n阈值建议：")
    print(f"当前阈值 4.0 {'合理' if percentiles[0] <= 4.0 <= percentiles[2] else '可能需要调整'}")
    if 4.0 < percentiles[0]:
        print(f"阈值 4.0 低于 25% 分位数 ({percentiles[0]:.3f})，可能过于严格，建议尝试 {percentiles[1]:.3f} 或 {percentiles[2]:.3f}")
    elif 4.0 > percentiles[2]:
        print(f"阈值 4.0 高于 75% 分位数 ({percentiles[2]:.3f})，可能引入过多边，建议尝试 {percentiles[0]:.3f} 或 {percentiles[1]:.3f}")

def main():
    parser = argparse.ArgumentParser(description="分析 PDB 文件中原子间距离分布以验证阈值")
    parser.add_argument('--config', type=str, default='../config.yaml', help='配置文件路径')
    parser.add_argument('--output_dir', type=str, default='./analysis/distance', help='分析结果输出目录')
    args = parser.parse_args()

    # 加载配置文件
    config = load_config(args.config)

    # 从配置文件中提取路径和参数
    csv_path = '../datasets/lastfinal_cyclic_peptides.csv'
    pdb_dir = '../pdb_files'
    max_len = config['data']['max_len']

    # 执行距离分布分析
    analyze_distance_distribution(csv_path, pdb_dir, max_len, args.output_dir)

if __name__ == "__main__":
    main()