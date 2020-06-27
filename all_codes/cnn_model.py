from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import initializers
import tensorflow as tf
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import Adam

def create_model(seq_type):
    
    # Structure Module
                                    #layer 1
    inp = Input(shape=(1, 60, 64))
    out = Conv2D(64,(3, 3), padding="same", kernel_initializer=initializers.glorot_uniform(seed=0),
         kernel_regularizer=l2(0.001), 
                 use_bias=False
    )(inp)
    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    out = Dropout(0.7)(out)
    out = MaxPooling2D((2, 2))(out)
                                    #layer 2
    out = Conv2D(128, (3, 3),padding="same", kernel_initializer=initializers.glorot_uniform(seed=0),
        kernel_regularizer=l2(0.001), 
                 use_bias=False
    )(out)
    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    out = Dropout(0.7)(out)
    out = MaxPooling2D((2, 2))(out)
                                    #layer3
    out = Conv2D(256, (3, 3),padding="same", kernel_initializer=initializers.glorot_uniform(seed=0),
        kernel_regularizer=l2(0.001), 
                 use_bias=False
    )(out)
    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    out = Dropout(0.7)(out)
    
    out = Flatten()(out)
    out = Dense(1024, kernel_initializer=initializers.glorot_uniform(seed=0),
                use_bias=False,
               activation="relu"
               )(out)
    out = Dropout(0.7)(out)

    # Sequence Module
    if seq_type == 'emb':
        inp_seq = Input(shape=(9))
        inp_out = Embedding(21, 20)(inp_seq)
        out1 = Conv1D(64,3,padding="same",activation="relu",
                  kernel_initializer=initializers.glorot_uniform(seed=0), use_bias=False,
                 kernel_regularizer=l2(0.001)
                 )(inp_out)
    else:
        inp_seq = Input(shape=(9,20))
        out1 = Conv1D(64,3,padding="same",activation="relu",
                  kernel_initializer=initializers.glorot_uniform(seed=0), use_bias=False,
                 kernel_regularizer=l2(0.001)
                 )(inp_seq)

    out1 = BatchNormalization()(out1)
    out1 = Activation("relu")(out1)
    out1 = Dropout(0.7)(out1)    
    out1 = Conv1D(128,3,padding="same",activation="relu",
                  kernel_initializer=initializers.glorot_uniform(seed=0), use_bias=False, 
                 kernel_regularizer=l2(0.001)
                 )(out1)
    out1 = BatchNormalization()(out1)
    out1 = Activation("relu")(out1)
    out1 = Dropout(0.7)(out1)
    out1 = Conv1D(256,3,padding="same",activation="relu",
                  kernel_initializer=initializers.glorot_uniform(seed=0), use_bias=False,
                 kernel_regularizer=l2(0.001)
                 )(out1)
    out1 = BatchNormalization()(out1)
    out1 = Activation("relu")(out1)
    out1 = Dropout(0.7)(out1)
    out1 = MaxPool1D(3)(out1)
    
    #out1 = Dropout(dropout)(out1)
    out1 = Flatten()(out1)
    out1 = Dense(1024, kernel_initializer=initializers.glorot_uniform(seed=0), 
                 activation="relu", use_bias=False
                )(out1)
    out1 = Dropout(0.7)(out1)
    # Combining
    out = Concatenate()([out,out1])  # comment out for switching
    
    out = Dense(512, kernel_initializer=initializers.glorot_uniform(seed=0), 
                activation="relu", use_bias=False
               )(out)  # change input here for switching
    out = Dropout(0.5)(out)
    
    out = Dense(1, activation="sigmoid", use_bias=False)(out)
    model = Model(
        inputs=  [inp, inp_seq], outputs=out
    )  # change the input for switching between seq and struc
    model.compile(
        loss="mean_squared_error",
        #loss="binary_crossentropy",
        optimizer=Adam(lr=0.0001),
        metrics=['mse']
        #metrics=['accuracy',tf.metrics.AUC()]
    )
    return model
