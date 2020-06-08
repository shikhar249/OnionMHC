##
import argparse
import data
import encode as enc
import evaluation
import importlib
#from cnn_model import *  # for model type
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedShuffleSplit, ShuffleSplit
import random
import os
import sys
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
K.set_image_data_format("channels_first")
print("Modules Imported")
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
print("Loading Data...\n")
df_features = data.features() #for sequence types
mod_ic, target = data.ic(), data.labels()
print("Data Loaded\n")
sc = StandardScaler()
sc.fit(df_features)
df_features = sc.transform(df_features)
df_features = df_features.reshape(-1,1,60,64)

earlystopper = callbacks.EarlyStopping(
    monitor="val_loss", mode='min',
    min_delta=0.0001,
    patience=30,
    verbose=1,
    restore_best_weights=True,
)
all_models = []
all_result = []

#defining training and validation data
#whole_sss = StratifiedShuffleSplit(n_splits=1, random_state=0, test_size=0.1)  		#change for classifier or regressor
whole_sss = ShuffleSplit(n_splits=1, random_state=0, test_size=0.1)
for i, (train_index, test_index) in enumerate(whole_sss.split(df_features, mod_ic)):		#change here  
    whole_tr=train_index
    whole_ts=test_index
print("Split Created\n")

comb_whole_model = []
for i in range(3):
    print("="*100)
    print("Model {} Training...".format(i))
    model = None
    model = create_model(args.s)
    result=model.fit([df_features[whole_tr], df_sequences[whole_tr]],
                mod_ic[whole_tr],
                #target[whole_tr], 
                batch_size= 64,   #change tar to ic or ic to tar
                epochs=800,
                validation_data = ([df_features[whole_ts], df_sequences[whole_ts]],
                mod_ic[whole_ts]),
                #target[whole_ts]),
                shuffle=True,
                verbose=1,
                callbacks = [earlystopper]
                )
    comb_whole_model.append(model)
    model.save('models/regressor/whole_model{}_{}_{}.h5'.format(i, args.s, args.m))
    print("Model {} trained and saved".format(i))
##
# benchmark testing
print("benchmark evaluation")
#evaluation.bench_eval(comb_whole_model)

##

