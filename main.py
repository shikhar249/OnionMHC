##
import argparse
import data
import encode as enc
import evaluation
import importlib
from cnn_model import *  # change model_type
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import random
import os
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
K.set_image_data_format("channels_first")
print("\nModules Imported")
##
# fetch data
os.chdir('/home/shikhar/peptide_proj/model_files/')

parser = argparse.ArgumentParser()
parser.add_argument('-s', required=True, help="Sequence encoding: emb / one / bls")
parser.add_argument('-m', required=True, help="model type: cnn / lstm")

args = parser.parse_args()

if args.m == "cnn":
    from cnn_model import *
elif args.m == "lstm":
    from lstm_model import *
else:
    print("specify model type: cnn / lstm")
    sys.exit(1)

if args.s == "emb":
    df_sequences = enc.embed(data.sequences())
elif args.s == "one":
    df_sequences = enc.one_hot(data.sequences())
elif args.s == "bls":
    df_sequences = enc.blosum_encode(data.sequences())
else:
    print("specify correct sequences type: emb / one / bls")
    sys.exit(1)

##
print("\nLoading Data...")
df_features = data.features() #, enc.embed(data.sequences()) #for sequence types
mod_ic, target = data.ic(), data.labels()
print("\nData Loaded")
sc = StandardScaler()
sc.fit(df_features)
df_features = sc.transform(df_features)
df_features = df_features.reshape(-1,1,60,64)

#skf = StratifiedKFold(n_splits=5, random_state=0)
skf = KFold(n_splits = 5, random_state=0)
tr, ts = [], []

for i, (train_index, test_index) in enumerate(skf.split(df_features, mod_ic)):
    tr.append(train_index)
    ts.append(test_index)

earlystopper = callbacks.EarlyStopping(
    monitor="val_loss", mode='min',
    min_delta=0.0001,
    patience=30,
    verbose=1,
    restore_best_weights=True,
)
for i in range(5):
    random.shuffle(tr[i])
    random.shuffle(ts[i])
print("\nFolds Created")
all_models = []
all_result = []
for i in range(5):
    df_train, df_test = df_features[tr[i]], df_features[ts[i]]
    seq_train, seq_test = df_sequences[tr[i]], df_sequences[ts[i]]
    tar_train, tar_test = target[tr[i]], target[ts[i]]
    ic_train, ic_test = mod_ic[tr[i]], mod_ic[ts[i]]
    X_train, X_test = [df_train, seq_train], [df_test, seq_test]
    fold_model = []
    for j in range(3):
        print("="*100)
        print("\nStart training fold {} model {}...\n".format(i,j))
        model = None
        model = create_model(args.s)

        result=model.fit(X_train,ic_train, batch_size= 64,   #change tar to ic or ic to tar
            epochs=600,
            validation_split = 0.1,
            shuffle=True,
            verbose=1,
            callbacks = [earlystopper]
            )
        model.save('/home/shikhar/peptide_proj/model_files/models/fold{}_model{}_{}_{}.h5'.format(i, j, args.s, args.m))
        all_result.append(result)
        print("\nTraining completed for fold {} model {}\n".format(i,j))
        print("+"*100)

##
# benchmark testing
#evaluation.bench_eval(all_models,sc)

##

