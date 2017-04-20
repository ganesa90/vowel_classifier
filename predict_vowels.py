# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 16:45:38 2017

@author: ganesh
"""

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

DATADIR = '/home/ganesh/research/vowel_classifier/data/clip_nsp_stimuli/'
featDIR = DATADIR+'/seg_mfcc/'
vowels = ['aa', 'ae', 'ah',  'eh',  'ey', 'ih', 'iy', 'ow',  'uh', 'uw']
model = load_model('vowel_classifier_seg_mfcc_dnn_512_512_nsp_bestmodel.h5')
with open('filelist_clip_nsp_mfcc.lst') as fp:
    flist = fp.read().splitlines()
for fpath in flist:
    fnm= os.path.basename(fpath).split('.')[0]
    feat = np.load(featDIR+fnm+'.npy')
    out = model.predict(feat.T)
    lab_indx = np.argmax(out)
    label = vowels[lab_indx]
    print(fnm+" "+label)

