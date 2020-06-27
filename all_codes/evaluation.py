from sklearn.metrics import (roc_auc_score, precision_recall_curve,
    auc, accuracy_score, f1_score)
from scipy.stats import pearsonr, spearmanr
import benchmark
import tensorflow as tf
import numpy as np

def evaluation(y_true, y_pred):
    #emb = tf.convert_to_tensor(emb, dtype=tf.float64)
    roc_curve = roc_auc_score(list(y_true), list(y_pred))
    lr_precision, lr_recall, _ = precision_recall_curve(y_true, y_pred)
    prc_curve = auc(lr_recall, lr_precision)
    acc = accuracy_score(y_true, np.array([int(x > 0.5) for x in y_pred]))
    f1 = f1_score(y_true, np.array([int(x > 0.5) for x in y_pred]))
    pcc = pearsonr(y_true, y_pred)
    scc = spearmanr(y_true, y_pred)
    return {'accuracy':acc, 'AUC':roc_curve, 'AUPRC':prc_curve, 'F1-score':f1,
            'pearson': pcc, 'spearman': scc}

#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) 

def bench_eval(model, sc, seq_int, metric_):
    bench_list = ['bench1', 'bench2', 'bench3', 'bench4', 'bench5', 'bench6', 'bench7', 'bench8',
                  'bench_ic1', 'bench_ic2', 'bench_ic3', 'bench_ic4', 'bench_ic5', 'bench_ic6', 'bench_ic7',
                  'bench_ic9', 'bench_t12_1','bench_t12_3']#,'bench_t12_2','bench_ic8']
    bench_res = []
    seq_dict = {0: 'emb', 1: 'enc', 2: 'bls'}
    for bench in bench_list:
        struc, y_true, ic = benchmark.struc_lab(bench)
        seq = benchmark.seq_inf(seq_dict[seq_int], bench)
        seq = tf.convert_to_tensor(seq, dtype=tf.float64)
        struc = sc.transform(struc)
        struc = struc.reshape(-1, 1, 60, 64)
        if type(model) != type([]):
        #If it's not an ensemble
            y_pred = np.squeeze(model.predict([struc,seq]))
        else:
        #Otherwise
            probas_ = [np.squeeze(m.predict([struc, seq])) for m in model]
            y_pred = [np.mean(scores) for scores in zip(*probas_)]
        y_pred_list = [int(x>0.5) for x in y_pred]
        #y_true = [int(i>0.4256) for i in ic]
        if metric_ == 'sp':
            score = abs(spearmanr(ic, y_pred)[0])
        if metric_ == 'auc':
            score = roc_auc_score(y_true, y_pred)
        if metric_ == 'f1':
            score = f1_score(y_true, np.array([int(x > 0.5) for x in y_pred]))
        print(score)
        bench_res.append(score)
    return np.mean(bench_res)

