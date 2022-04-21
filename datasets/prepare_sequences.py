import h5py
from tqdm import tqdm
import librosa
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle
import cv2

classes13 = {
  "Pipistrellus pipistrellus": 0,
  "Pipistrellus nathusii": 1,
  "Pipistrellus kuhlii": 2,
  "Myotis daubentonii": 3,
  "Nyctalus noctula": 4,
  "Nyctalus leisleri": 5,
  "Eptesicus serotinus": 6,
  "Myotis dasycneme": 7,
  "Miniopterus schreibersii": 8,
  "Vespertilio murinus": 9,
  "Rhinolophus ferrumequinum": 10,
  "Myotis emarginatus": 11,
  "Myotis myotis": 12,
}

classes18 = {
  "Pipistrellus pipistrellus": 0,
  "Pipistrellus nathusii": 1,
  "Pipistrellus kuhlii": 2,
  "Myotis daubentonii": 3,
  "Nyctalus noctula": 4,
  "Nyctalus leisleri": 5,
  "Myotis nattereri": 6,
  "Eptesicus serotinus": 7,
  "Myotis dasycneme": 8,
  "Miniopterus schreibersii": 9,
  "Vespertilio murinus": 10,
  "Rhinolophus ferrumequinum": 11,
  "Myotis brandtii": 12,
  "Myotis mystacinus": 13,
  "Myotis emarginatus": 14,
  "Myotis myotis": 15,
  "Eptesicus nilssonii": 16,
  "Rhinolophus blasii": 17
}

def slideWindow(a, size, step, resize):
    b = []
    i = 0
    pos = 0
    while pos + size < len(a):
        pos = int(i * step)
        tile = a[pos : pos + size]
        if resize is not None:
            tile = cv2.resize(tile, dsize=resize, interpolation=cv2.INTER_NEAREST)
        b.append(tile)
        i+=1
    return b

def getSequences(spectrogram, patch_len, patch_skip, seq_len, seq_skip, resize):
    tiles = slideWindow(spectrogram, patch_len, patch_skip, resize)[:-1] # last one is not full
    sequences = slideWindow(tiles, seq_len, seq_skip, None)[:-1] # last one is not full
    return sequences

def prepareSet(prepared_set, labels, patch_len, patch_skip, seq_len, seq_skip, resize, one_hot):
    X_seq = []
    Y_seq = []
    
    for species in tqdm(list(labels)):
        S_db = prepared_set.get(species)
        label = to_categorical(labels[species], num_classes=len(labels)) if one_hot else labels[species]

        seq = getSequences(S_db, patch_len, patch_skip, seq_len, seq_skip, resize)
        X_seq.extend(seq)
        Y_seq.extend([label] * len(seq))
    
    X_seq, Y_seq = shuffle(X_seq, Y_seq, random_state=42)
    return np.asarray(X_seq), np.asarray(Y_seq)

def prepare(file, labels, patch_len, patch_skip, seq_len, seq_skip, resize=None, one_hot=False):
    prepared_hf = h5py.File(file, 'r')
    X_train, Y_train = prepareSet(prepared_hf.require_group("train"), labels, patch_len, patch_skip, seq_len, seq_skip,
                                  resize, one_hot)
    X_test, Y_test = prepareSet(prepared_hf.require_group("test"), labels, patch_len, patch_skip, seq_len, seq_skip,
                                resize, one_hot)
    X_val, Y_val = prepareSet(prepared_hf.require_group("val"), labels, patch_len, patch_skip, seq_len, seq_skip,
                              resize, one_hot)
    return X_train, Y_train, X_test, Y_test, X_val, Y_val