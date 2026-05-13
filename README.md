# MFCPepPred File Descriptions

- `datasets`--Stores data
  - `lastfinal_cyclic_peptides.csv`--Cyclic peptide data

- `img`--Stores images
  - `upset.tif`--Upset plot corresponding to the data

- `onlySmiles`--Only uses large language models
  - `chemberta_multi_label_final`、`pubchem_multi_label_final`--Saved best models

- `test`folder--Used to store testing code
  - `tokensTest.py`--Tests the encoding method of the `ChemBERTa-77M-MLM` model
- `dataset.py`--Data processing
- `evaluate.py`--Evaluation metrics
-  `generate3DMultithreaded.py`--Construct 3D structures and obtain PDB files
- `loss.py`--Loss functions
- `main.py`--Main program
- `model.py`--Model
- `train.py` -- Training model
- `test.py` -- Model evaluation on test set
- `sampling.py` -- Hard negative/positive mining
- `imbalance_utils.py` -- Label statistics and weighted sampler
- `utils.py` -- Seed setting, config loading, collate function
- `config.yaml` -- All hyperparameters and paths

## Create and activate environment

```bash
conda env create -f environment.yml
conda activate CP
```


## Training 

The model relies on 3D atomic coordinates. To generate them in advance:

```
python generate3DMultithreaded.py
```

To train a model for prediction, you can run:

```
python main.py
```
