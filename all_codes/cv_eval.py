import argparse
import random
import data
import encode as enc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, Kfold
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('-s', required=True, help="Sequence encoding: emb / one / bls")
#parser.add_argument('-m', required=True, help="model type: cnn / lstm")

args = parser.parse_args()

#if args.m == "cnn":
#    from cnn_model import *
#elif args.m == "lstm":
#    from lstm_model import *
#else:
#    print("specify model type: cnn / lstm")
#    sys.exit(1)

if args.s == "emb":
    df_sequences = enc.embed(data.sequences())
elif args.s == "one":
    df_sequences = enc.one_hot(data.sequences())
elif args.s == "bls":
    df_sequences = enc.blosum_encode(data.sequences())
else:
    print("specify correct sequences type: emb / one / bls")
    sys.exit(1)

df_features = data.features()  # enc.blosum_encode(data.sequences()).astype('float64') #for sequence types
mod_ic, target = data.ic(), data.labels()

sc = StandardScaler()
sc.fit(df_features)
df_features = sc.transform(df_features)
df_features = df_features.reshape(-1,1,60,64)

#skf = StratifiedKFold(n_splits=5, random_state=0)
skf = Kfold(n_splits=5, random_state=0)
tr, ts = [], []

for i, (train_index, test_index) in enumerate(skf.split(df_features, mod_ic)): #change for regressor
    tr.append(train_index)
    ts.append(test_index)

for i in range(5):
    random.shuffle(tr[i])
    random.shuffle(ts[i])


for i in range(5):
    cv_res = []
    #print(df_features[ts[i]])
    #print(df_sequences[ts[i]])
    for j in range(3):
        model = None
        model = load_model('models/fold{}_model{}.h5'.format(i,j))
        pred_ = model.predict([df_features[ts[i]], df_sequences[ts[i]]])
        cv_res.append(roc_auc_score(target[ts[i]], pred_))
        del model
    print('Fold ',i,' AUC scores:', cv_res[0], cv_res[1], cv_res[2])
