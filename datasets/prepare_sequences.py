import h5py
from tqdm import tqdm
import librosa
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle
import cv2


germanBats = {
    "Rhinolophus ferrumequinum": 0,
    "Rhinolophus hipposideros": 1,
    "Myotis daubentonii": 2,
    "Myotis brandtii": 3,
    "Myotis mystacinus": 4,
    "Myotis emarginatus": 5,
    "Myotis nattereri": 6,
    #"Myotis bechsteinii": 7,
    "Myotis myotis": 7,
    "Myotis dasycneme": 8,
    "Nyctalus noctula": 9,
    "Nyctalus leisleri": 10,
    "Pipistrellus pipistrellus": 11,
    "Pipistrellus nathusii": 12,
    "Pipistrellus kuhlii": 13,
    "Eptesicus serotinus": 14,
    "Eptesicus nilssonii": 15,
    #"Plecotus auritus": 16,
    #"Plecotus austriacus": 16,
    #"Barbastella barbastellus": 16,
    #"Tadarida teniotis": 16,
    "Miniopterus schreibersii": 16,
    #"Hypsugo savii": 18,
    "Vespertilio murinus": 17,
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

def getSequences(spectrogram, patch_len, patch_skip, options, mode, resize):
    tiles = slideWindow(spectrogram, patch_len, patch_skip, resize)[:-1] # last one is not full
    if mode == 'slide':
        seq_len = options['seq_len']
        seq_skip = options['seq_skip']
        sequences = slideWindow(tiles, seq_len, seq_skip, None)[:-1] # last one is not full
        return sequences
    
    elif mode == 'pick_random':
        min_len = options['min_len']
        max_len = options['max_len']
        overlap_coeff = options['overlap_coeff']
        length = len(tiles)
        num_seqs = int((overlap_coeff * length) / ((min_len + max_len) / 2))
        sequences = []
        for i in range(num_seqs):
            seq_len = np.random.randint(min_len, max_len + 1)
            seq_start = np.random.randint(0, length - seq_len)
            seq = np.zeros((max_len, patch_len, tiles[0].shape[-1]))
            for k in range(0, seq_len):
                seq[k] = tiles[seq_start + k]
            sequences.append(seq)
        return sequences

def prepareSet(prepared_set, labels, patch_len, patch_skip, options, mode, resize, one_hot):
    X_seq = []
    Y_seq = []
    
    for species in tqdm(list(labels)):
        S_db = prepared_set.get(species)
        label = to_categorical(labels[species], num_classes=len(labels)) if one_hot else labels[species]

        seq = getSequences(S_db, patch_len, patch_skip, options, mode, resize)
        X_seq.extend(seq)
        Y_seq.extend([label] * len(seq))
    
    X_seq, Y_seq = shuffle(X_seq, Y_seq, random_state=42)
    return np.asarray(X_seq), np.asarray(Y_seq)

def prepare(file, labels, patch_len, patch_skip, options, mode='slide', resize=None, one_hot=False, only_test=False):
    prepared_hf = h5py.File(file, 'r')
    if only_test:
        X_test, Y_test = prepareSet(prepared_hf.require_group("test"), labels, patch_len, patch_skip, options, mode, resize, one_hot)
        return X_test, Y_test
    
    X_train, Y_train = prepareSet(prepared_hf.require_group("train"), labels, patch_len, patch_skip, options, mode, resize, one_hot)
    X_test, Y_test = prepareSet(prepared_hf.require_group("test"), labels, patch_len, patch_skip, options, mode, resize, one_hot)
    X_val, Y_val = prepareSet(prepared_hf.require_group("val"), labels, patch_len, patch_skip, options, mode, resize, one_hot)
    return X_train, Y_train, X_test, Y_test, X_val, Y_val