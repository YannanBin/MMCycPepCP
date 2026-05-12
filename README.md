# MMCycPepCP File Descriptions

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
- `generate3DMultithreaded.py`--Uses multithreading to construct 3D structures, obtains 5 conformations, and selects the best one (the one with the minimum energy value)
- `loss.py`--Loss functions
- `main.py`--Main program
- `model.py`--Model




