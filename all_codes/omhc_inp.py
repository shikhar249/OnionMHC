import numpy as np
import pandas as pd
import encode as enc
from math import log
import numpy as np


def from_ic50(ic50, max_ic50=50000.0):
    x = 1.0 - (np.log(np.maximum(ic50, 1e-12)) / np.log(max_ic50))
    return np.minimum( 1.0, np.maximum(0.0, x))

def struc_lab(struc_file):

    feat = pd.read_csv(struc_file)
    feat = np.array(feat.drop(feat.columns[0], axis=1))
    
    return feat 

def ic_labels(struc_file):
    
    labels = np.array(pd.read_csv(struc_file + "bench_label", header=None).iloc[:, 0])
    ic50 =  np.array(pd.read_csv(struc_file + "bench_ic50", header=None).iloc[:, 0])
    
    return labels, ic50


def seq_inf(typed, seq_file):
    
    bench_sequences = list(pd.read_csv(seq_file, header=None).iloc[:, 0])
    if typed == 'enc':
        return enc.one_hot(bench_sequences)
    if typed == 'bls':
        return enc.blosum_encode(bench_sequences)
    if typed == 'emb':
        return enc.embed(bench_sequences)

