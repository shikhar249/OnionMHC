import argparse
from argparse import RawTextHelpFormatter
import data
from evaluation import bench_eval
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import os, sys
#import tensorflow as tf

#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
parser = argparse.ArgumentParser(description='Evaluating the trained models on benchmark datasets...', formatter_class=RawTextHelpFormatter)
parser.add_argument('-seq', type=int, help="0: embedding, 1: one_hot, 2: blosum", default=-1)
parser.add_argument('-mod', nargs="*", help="Models to evaluate (*.h5)")
parser.add_argument('-met', default='auc',
                            help= "sp: spearman correlation\n"
                                  "auc: area under ROC curve\n"
                                  "f1: F1-score\n")
args = parser.parse_args()


if args.seq not in (0,1,2):  
    print("\n-seq takes only 0, 1 or 2 as argument. Check --help\n")
    sys.exit(1)

df_features = data.features()
sc = StandardScaler()
sc.fit(df_features)
                    
def evaluat(abc, seq, met):
    all_models = []

    for mod in abc:
        all_models.append(load_model(mod))
    
    print("Average AUC:", bench_eval(all_models,sc,seq, met))
    
evaluat(args.mod, args.seq, args.met)

