import argparse
from argparse import RawTextHelpFormatter
import data
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import os, sys
import omhc_inp
import tensorflow as tf
import numpy as np
from sklearn.metrics import (roc_auc_score, precision_recall_curve,
    auc, accuracy_score, f1_score)
from scipy.stats import pearsonr, spearmanr
import pandas as pd
from tensorflow.keras.models import model_from_json

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
parser = argparse.ArgumentParser(description='Making Predictions using OnionMHC', formatter_class=RawTextHelpFormatter)
parser.add_argument('-struc', type=str, help="File containing structure based features in .csv format as produced by \"generate_features_cam_o.py\"")
parser.add_argument('-seq', type=str, help="File containing the sequence of 9-mer peptides")
parser.add_argument('-out', help="Output file to save results to ")

args = parser.parse_args()

df_features = data.features()
sc = StandardScaler()
sc.fit(df_features)


def evaluat( seq_int, sc):

    seq_dict = {0: 'emb', 1: 'enc', 2: 'bls'}
    struc = omhc_inp.struc_lab(args.struc)
    seq = omhc_inp.seq_inf(seq_dict[seq_int], args.seq)
    seq = tf.convert_to_tensor(seq, dtype=tf.float64)
    struc = sc.transform(struc)
    struc = struc.reshape(-1, 1, 60, 64)

    for i in range(5):
        for j in range(3):
            json_f = open("/home/shikhar/peptide_proj/model_files/models/cross_val/fold" + str(i) + "_model" + str(j) + "_bls_lstm.json", 'r')
            loaded_model_json = json_f.read()
            json_f.close()
            model = model_from_json(loaded_model_json)
            model.load_weights(("/home/shikhar/peptide_proj/model_files/models/cross_val/fold" + str(i) + "_model" + str(j) + "_bls_lstm_weights.h5"))
            y_pred = list(np.squeeze(model.predict([struc, seq])))
            if i == 0 and j == 0:
                bench_ = y_pred
            else:
                bench_ = [sum(x) for x in zip(bench_, y_pred)]

            del model



    bench_ = [x/15 for x in bench_]
    
    return bench_


output = evaluat( 2, sc)
omhc_bf = [50000**(1-x) for x in output]
sequences = list(pd.read_csv(args.seq, header=None).iloc[:,0])

d = {'peptide_sequences': sequences, 'OnionMHC_score': output, 'Binding_Affinity(nM)':omhc_bf}

output_df = pd.DataFrame(data=d)


if args.out:
    output_df.to_csv(args.out, sep='\t', index=False, float_format="%.4f")
else:
    print(output_df)

