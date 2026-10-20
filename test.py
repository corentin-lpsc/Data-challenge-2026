import pandas as pd

#importattion des tableaux
data_mol_train = pd.read_csv("molecular_train.csv")
data_mol_test  = pd.read_csv("molecular_test.csv")
data_cli_train = pd.read_csv("clinical_train.csv")
data_cli_test  = pd.read_csv("clinical_test.csv")