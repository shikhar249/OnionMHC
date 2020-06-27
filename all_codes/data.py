import pandas as pd
import numpy as np
from math import log

path = "/home/shikhar/peptide_proj/all0201_sequences/"

def from_ic50(ic50, max_ic50=50000.0):
    x = 1.0 - (np.log(np.maximum(ic50, 1e-12)) / np.log(max_ic50))
    return np.minimum( 1.0, np.maximum(0.0, x))

def features():
    df_features = pd.read_csv(path+"9051_peptides.csv")
    df_features = np.array(df_features.drop(df_features.columns[0], axis=1))
    return df_features


def sequences():
    df_sequences = pd.read_csv(path+"9051_ordered_seq_ic", sep='\t', header=None)
    seq = list(df_sequences.iloc[:, 0])
    return seq


def ic():
    df_sequences = pd.read_csv(path+"9051_ordered_seq_ic", sep='\t', header=None)
    ic = np.array(df_sequences.iloc[:, 2])
    ic_transformed = from_ic50(ic)
    return ic_transformed


def labels():
    target = np.array(pd.read_csv(path+"9051_labels", header=None))
    return target

