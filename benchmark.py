import numpy as np
import pandas as pd
import encode as enc

path = "/home/shikhar/scwrl4/md_struc/all_em_seq/benchmark/"
bench_dir = ["bench1_01022019/", "bench2_19022016/", "bench3_19062015/", "bench4_20062014/", "bench5_23052014/",
             "bench6_28032014/", "bench7_21032014_1026840/", "bench8_21032014_1026941/", "bench9_ic_15032019/", "bench10_ic_01022019/", "bench11_ic_11052018/",
             "bench12_ic_15052015/","bench13_ic_06022015_1028553/", "bench14_ic_06022015_1028554/", "bench15_ic_16012015/", "bench16_ic_21032014_1026840/","bench17_ic_21032014_1026941/",
             "bench18_t_half_1026371_21032014/","bench19_t_half_1026840_21032014/", "bench20_t_half_1028285_21032014/"]

diff_benches = ['bench1', 'bench2', 'bench3', 'bench4', 'bench5', 'bench6', 'bench7', 'bench8', 'bench_ic1',
                'bench_ic2', 'bench_ic3', 'bench_ic4', 'bench_ic5', 'bench_ic6', 'bench_ic7', 'bench_ic8','bench_ic9', 'bench_t12_1', 'bench_t12_2', 'bench_t12_3']

def struc_lab(bench_type):

    ind = diff_benches.index(bench_type)

    feat = pd.read_csv(path + bench_dir[ind] +"output.csv")
    feat = np.array(feat.drop(feat.columns[0], axis=1))

    labels = np.array(pd.read_csv(path + bench_dir[ind] + "bench_label", header=None).iloc[:, 0])
    ic50 =  np.array(pd.read_csv(path + bench_dir[ind] + "bench_ic50", header=None).iloc[:, 0])
    return feat, labels, ic50


def seq_inf(typed, bench_type):
    ind = diff_benches.index(bench_type)
    bench_sequences = list(pd.read_csv(path + bench_dir[ind] + "bench_seq", header=None).iloc[:, 0])
    if typed == 'enc':
        return enc.one_hot(bench_sequences)
    if typed == 'bls':
        return enc.blosum_encode(bench_sequences)
    if typed == 'emb':
        return enc.embed(bench_sequences)

