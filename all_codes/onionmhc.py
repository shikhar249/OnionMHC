import argparse
from argparse import RawTextHelpFormatter
import data
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import os, sys
import benchmark
import tensorflow as tf
import numpy as np
from sklearn.metrics import (roc_auc_score, precision_recall_curve,
    auc, accuracy_score, f1_score)
from scipy.stats import pearsonr, spearmanr
import pandas as pd

#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
parser = argparse.ArgumentParser(description='Evaluating the trained models on benchmark datasets...', formatter_class=RawTextHelpFormatter)
parser.add_argument('-struc', type=str, help="File containing structure based features in .csv format as produced by \"generate_features_cam_o.py\"")
parser.add_argument('-seq', type=str, help="File containing the sequence of 9-mer peptides")
parser.add_argument('-mod', nargs="*", help="Models to evaluate (*.h5)")
parser.add_argument('-out', help="Output file to save results to ")

args = parser.parse_args()

df_features = data.features()
sc = StandardScaler()
sc.fit(df_features)


fol = list(pd.read_csv("folders", header=None).iloc[:,0])

def evaluat(abc, seq_int, sc):

    sequences = list(pd.read_csv(args.seq, header=None).iloc[:,0])
    bench_ = []
    seq_dict = {0: 'emb', 1: 'enc', 2: 'bls'}
    print(len(abc))
    struc = benchmark.struc_lab(args.struc)
    seq = benchmark.seq_inf(seq_dict[seq_int], args.seq)
    seq = tf.convert_to_tensor(seq, dtype=tf.float64)
    struc = sc.transform(struc)
    struc = struc.reshape(-1, 1, 60, 64)
    model = load_model(abc[0])

    y_pred = list(np.squeeze(model.predict([struc, seq])))
    bench_ = y_pred
    del model
    print(bench_)

    for mod in abc[1:]:
        model = load_model(mod)
        print(mod, "loaded")
        y_pred = list(np.squeeze(model.predict([struc, seq])))
        bench_ = [sum(x) for x in zip(bench_, y_pred)]

        del model

    bench_ = [x/len(abc) for x in bench_]
    
    print(bench_)
    
    return bench_

 
output = evaluat(args.mod, 2, sc)
omhc_bf = [50000**(1-x) for x in output]
sequences = list(pd.read_csv(args.seq, header=None).iloc[:,0])

d= {'peptide_sequences': sequences, 'OnionMHC_score': output, 'Binding_Affinity(nM)':omhc_bf}

output_df = pd.DataFrame(data=d)

output_df.to_csv(args.out, sep='\t', index=False, float_format="%.4f")
