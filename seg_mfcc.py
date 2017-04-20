import numpy as np
import HTK
from HTK import *
import os
from sklearn.preprocessing import *
import librosa


def prepare_data():
    DATADIR = '/home/ganesh/research/vowel_classifier/data/clip_nsp_stimuli/'  
    featDIR = DATADIR+'/mfcc/'
    opdir = DATADIR+'/seg_mfcc/'
    vowels = {}
    vowels = {'aa':0, 'ae':1, 'ah':2,  'eh':3,  'ey':4, 'ih':5, 'iy':6, 'ow':7,  'uh':8, 'uw':9}
    if not os.path.exists(opdir):
        os.makedirs(opdir)
    with open('filelist_clip_nsp_mfcc.lst') as fp:
        flist = fp.read().splitlines()
    for fpath in flist:
        fnm= os.path.basename(fpath).split('.')[0]
        hfile = HTKFile()
        if not os.path.exists(featDIR+fnm+'.htk'):
            continue
        hfile.load(featDIR+fnm+'.htk')
        feat = np.asarray(hfile.data)
        lenfeat = feat.shape[0]
        if lenfeat >= 3:
            seglen = int(np.floor(lenfeat/3))
            seg1 = np.mean(feat[0:seglen,:], axis=0)[None,:]
            seg2 = np.mean(feat[seglen:2*seglen,:], axis=0)[None,:]
            seg3 = np.mean(feat[2*seglen:,:], axis=0)[None,:]
            segfeat = np.concatenate((seg1, seg2, seg3), axis=1).T
        elif lenfeat == 2:
            segfeat = np.concatenate((feat[0,:], feat[1,:], feat[1,:]), axis = 1).T
        elif lenfeat == 1:
            segfeat = np.concatenate((feat[0,:], feat[0,:], feat[0,:]), axis = 1).T
        max_feat = np.max(feat, axis=0)[None,:]
        min_feat = np.min(feat, axis=0)[None,:]
        del_feat = librosa.feature.delta(feat, width=5, order=1, axis=0, trim=True)
        max_del_feat = np.max(del_feat, axis=0)[None,:]
        min_del_feat = np.min(del_feat, axis=0)[None,:]
        segfeat = np.concatenate((segfeat.T, max_feat, min_feat, max_del_feat, min_del_feat), axis=1).T
      #  labtxt = fpath.split('/')[7]
       # label = vowels[labtxt]
       # label_vec = np.zeros((10,1))
       # label_vec[label] = 1
        np.save(opdir+fnm+'.npy', segfeat)
       # np.save(opdir+fnm+'_lab.npy', label_vec)

if __name__ == "__main__":
    prepare_data()

