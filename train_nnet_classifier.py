import numpy as np
import HTK
import os
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from keras import backend as K
from save_object import *
import scipy
import scipy.io

def read_data(flist, featDIR):
    X = []
    Y = []
    for fpath in flist:
        fnm= os.path.basename(fpath).split('.')[0]
        feats = np.load(featDIR+fnm+'.npy')
        targ = np.load(featDIR+fnm+'_lab.npy')
        X.append(feats.T)
        Y.append(targ.T)
        if len(X)%1000 == 0:
            print(len(X))
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    return X,Y


def train_classifier():
    DATADIR = '/home/ganesh/research/vowel_classifier/data/nsp_vowels_dirs/'  
    featDIR = DATADIR+'/seg_mfcc/'
    opdir = './'
    with open('filelist_nsp_train.lst') as fp:
        flist = fp.read().splitlines()
    X_trn, Y_trn = read_data(flist, featDIR)
    with open('filelist_nsp_crsv.lst') as fp:
        flist = fp.read().splitlines()
    X_crv, Y_crv = read_data(flist, featDIR)
    with open('filelist_nsp_test.lst') as fp:
        flist = fp.read().splitlines()
    X_tst, Y_tst = read_data(flist, featDIR)
    print(X_trn.shape)
    X_mean = np.mean(X_trn, axis=0)
    X_std = np.std(X_trn, axis=0)
    X_trn = (X_trn - X_mean)/X_std
    X_tst = (X_tst - X_mean)/X_std
    X_crv = (X_crv - X_mean)/X_std

    print('Now building model')
    lnodes = [512, 512]
    nb_epoch = 200
    model = Sequential()
    model.add(Dense(lnodes[0], input_dim=X_trn.shape[1]))
    model.add(Activation('tanh'))
    for lnd in lnodes[1:]:
        model.add(Dense(lnd))
        model.add(Activation('tanh'))
    model.add(Dense(Y_trn.shape[1]))
    model.add(Activation('softmax'))
    # Defining the stopping criterion
    val_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
    weight_dims = [x.shape for x in model.get_weights()]
    layer_str = ''
    for i in range(1, len(weight_dims)-1, 2):
        layer_str = layer_str+"_"+str(weight_dims[i][0])

    filepath = opdir+'vowel_classifier_seg_mfcc_dnn'+layer_str+"_nsp_bestmodel.h5"
    save_bestmodel = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])

    print('Now training the model...')
    # Training the model
    model.fit(X_trn, Y_trn, batch_size=10000, nb_epoch=nb_epoch,
              verbose=2, validation_data=(X_crv, Y_crv), shuffle=True,
              callbacks=[val_stop, save_bestmodel])
    score = model.evaluate(X_tst, Y_tst, verbose=0)
    print(score)

if __name__ == "__main__":
    train_classifier()

