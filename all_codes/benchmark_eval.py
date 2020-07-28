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
                    
def evaluat(abc, seq_int, met, sc):
    #all_results = []
    #for num, mod in enumerate(abc):
    #    print("loading model ", mod)
    #    model = load_model(mod)
    #    all_results.append(bench_eval(model,sc,seq,met))

    #print(len(all_results))
    bench_list = ['bench1', 'bench2', 'bench3', 'bench4', 'bench5', 'bench6', 'bench7', 'bench8',
                  'bench_ic1', 'bench_ic2', 'bench_ic3', 'bench_ic4', 'bench_ic5', 'bench_ic6', 'bench_ic7',
                  'bench_ic9', 'bench_t12_1', 'bench_t12_3']
    bench_ =[]
    seq_dict = {0: 'emb', 1: 'enc', 2: 'bls'}
    print(len(abc))
    model = load_model(abc[0])
    for bench in bench_list:
        struc, _, _ = benchmark.struc_lab(bench)
        #print(bench, seq_dict[seq])
        seq = benchmark.seq_inf(seq_dict[seq_int], bench)
        seq = tf.convert_to_tensor(seq, dtype=tf.float64)
        struc = sc.transform(struc)
        struc = struc.reshape(-1, 1, 60, 64)
        y_pred = list(np.squeeze(model.predict([struc, seq])))
        bench_.append(y_pred)
    del model
    print(bench_[0][:10])

    for mod in abc[1:]:
        model = load_model(mod)
        print(mod, "loaded")
        for i, bench in enumerate(bench_list):
            struc, y_true, ic = benchmark.struc_lab(bench)
            seq = benchmark.seq_inf(seq_dict[seq_int], bench)
            seq = tf.convert_to_tensor(seq, dtype=tf.float64)
            struc = sc.transform(struc)
            struc = struc.reshape(-1, 1, 60, 64)
            y_pred = list(np.squeeze(model.predict([struc, seq])))
            bench_[i] = [sum(x) for x in zip(bench_[i], y_pred)]
            #if i == 0:
                #print(y_pred[:10])
                #print(bench_[0][:10])
        del model
    for i in range(len(bench_)):
        bench_[i] = [x/len(abc) for x in bench_[i]]
    #print(bench_[0][:10])
    #print("Average AUC:", bench_eval(all_models,sc,seq, met))

    for enum, bench in enumerate(bench_list):
        _, y_true, ic = benchmark.struc_lab(bench)
    #y_pred_list = [int(x>0.5) for x in y_pred]

        #if met == 'sp':
        sp = abs(spearmanr(ic, bench_[enum])[0])
        #if met == 'auc':
        auc = roc_auc_score(y_true, bench_[enum])
        if met == 'f1':
            score = f1_score(y_true, np.array([int(x > 0.5) for x in bench_[enum]]))
    #mod_.append(bench_)
        print(auc, sp)
    #print(np.mean(score))
    

#model = load_model(args.mod[0])

evaluat(args.mod, args.seq, args.met, sc)
