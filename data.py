import pandas as pd
import numpy as np
from math import log


def features():
    df_features = pd.read_csv("/home/shikhar/scwrl4/md_struc/all_em_seq/all0201_sequences/9051_peptides.csv")
    df_features = np.array(df_features.drop(df_features.columns[0], axis=1))
    return df_features


def sequences():
    df_sequences = pd.read_csv("/home/shikhar/scwrl4/md_struc/all_em_seq/all0201_sequences/9051_seq_ic", sep='\t', header=None)
    seq = list(df_sequences.iloc[:, 0])
    return seq


def ic():
    df_sequences = pd.read_csv("/home/shikhar/scwrl4/md_struc/all_em_seq/all0201_sequences/9051_seq_ic", sep='\t', header=None)
    ic = np.array(df_sequences.iloc[:, 2])
    ic_transformed = [[1 - log(x)/log(50000)] for x in ic]
    return np.array(ic_transformed)


def labels():
    target = np.array(pd.read_csv("/home/shikhar/scwrl4/md_struc/all_em_seq/all0201_sequences/9051_labels", header=None))
    return target

