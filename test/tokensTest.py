from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("../pre_model/DeepChem/ChemBERTa-77M-MLM")

# 对 SMILES 进行分词
smiles = "CC=C1NC(=O)C2CSSCCC=CC(CC(=O)NC(C(C)C)C(=O)N2)OC(=O)C(C(C)C)NC1=O"
tokens = tokenizer.tokenize(smiles)
print(tokens)

# 获取对应的 token IDs
inputs = tokenizer(smiles, return_tensors="pt")
print(inputs["input_ids"])