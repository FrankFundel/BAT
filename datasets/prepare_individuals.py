import h5py
from tqdm import tqdm
import librosa
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle
import cv2
import torch

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

def peak_detect(spectrogram, threshold):
    env = np.mean(spectrogram, axis=1)
    env[env < threshold] = 0
    peaks = librosa.util.peak_pick(env, pre_max=3, post_max=5, pre_avg=3, post_avg=5, delta=0.6, wait=20)
    return env, peaks

def getIndividuals(spectrogram, patch_len, resize=None, threshold=0):
    individuals = []
    _, peaks = peak_detect(spectrogram, threshold)
    for p in peaks:
        pos = p - int(patch_len / 2)
        if (pos >= 0 and len(spectrogram) >= pos+patch_len):
            ind = spectrogram[pos:pos+patch_len]
            if resize is not None:
                ind = cv2.resize(ind, dsize=resize, interpolation=cv2.INTER_NEAREST)
            individuals.append(ind)
    return individuals

def prepareSet(prepared_set, labels, patch_len, scale_factor, resize, one_hot, threshold):
    X_ind = []
    Y_ind = []

    for species in tqdm(list(labels)):
        S_db = np.asarray(prepared_set.get(species))
        new_size = (int(S_db.shape[1] * scale_factor), int(S_db.shape[0] * scale_factor))
        S_db = cv2.resize(S_db, dsize=new_size, interpolation=cv2.INTER_NEAREST)
        label = to_categorical(labels[species], num_classes=len(labels)) if one_hot else labels[species]

        ind = getIndividuals(S_db, patch_len, resize, threshold)
        X_ind.extend(ind)
        Y_ind.extend([label] * len(ind))
    
    X_ind, Y_ind = shuffle(X_ind, Y_ind, random_state=42)
    return np.asarray(X_ind), np.asarray(Y_ind)

def prepare(file, labels, patch_len, scale_factor=1.0, resize=None, one_hot=False, threshold=0):
    prepared_hf = h5py.File(file, 'r')

    X_train, Y_train = prepareSet(prepared_hf.require_group("train"), labels, patch_len, scale_factor,
                                  resize, one_hot, threshold)
    X_test, Y_test = prepareSet(prepared_hf.require_group("test"), labels, patch_len, scale_factor,
                                resize, one_hot, threshold)
    X_val, Y_val = prepareSet(prepared_hf.require_group("val"), labels, patch_len, scale_factor,
                              resize, one_hot, threshold)
    return X_train, Y_train, X_test, Y_test, X_val, Y_val