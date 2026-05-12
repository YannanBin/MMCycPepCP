# MMCycPepCP各文件说明

- `datasets`--存放数据
  - `lastfinal_cyclic_peptides.csv`--环肽的数据

- `img`--存放照片
  - `upset.tif`--数据对应的upset图

- `onlySmiles`--只使用了大语言模型
  - `chemberta_multi_label_final`、`pubchem_multi_label_final`--保存的最佳模型

- `test`文件夹--用来保存一些测试的代码
  - `tokensTest.py`--测试 `ChemBERTa-77M-MLM` 模型的编码方式
- `dataset.py`--数据处理
- `evaluate.py`--评价指标
- `generate3DMultithreaded.py`--使用多线程构建3D结构，得到5个构象择其优（能量值最小）
- `loss.py`--损失函数
- `main.py`--主程序
- `model.py`--模型




