import pandas as pd

# 读取两个文件
csv_path = "../datasets/lastfinal_cyclic_peptides.csv"
excel_path = "../datasets/ExportData.xlsx"

# 读取CSV文件（包含CPKB ID列）
csv_df = pd.read_csv(csv_path)

# 读取Excel文件（包含ID和Sequence列）
excel_df = pd.read_excel(excel_path)

# 合并数据（假设CSV中的CPKB ID对应Excel中的ID）
merged_df = pd.merge(
    csv_df,
    excel_df[['ID', 'Sequence']],
    left_on='CPKB ID',
    right_on='ID',
    how='left'
)

# 删除多余的ID列（如果不需要）
merged_df.drop('ID', axis=1, inplace=True, errors='ignore')

# 保存结果（可以覆盖原文件或新建文件）
output_path = "./updated_cyclic_peptides.csv"
merged_df.to_csv(output_path, index=False)

print(f"数据合并完成！结果已保存至: {output_path}")
print(f"新增了 {merged_df['Sequence'].notna().sum()} 条Sequence数据")
