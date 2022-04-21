import h5py
from tqdm import tqdm
import librosa
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle
import cv2
import torch

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

def peak_detect(spectrogram, threshold):
    env = np.mean(spectrogram, axis=1)
    env[env < threshold] = 0
    peaks = librosa.util.peak_pick(env, pre_max=3, post_max=5, pre_avg=3, post_avg=5, delta=0.6, wait=20)
    return env, peaks

def getIndividuals(spectrogram, patch_len, resize, threshold, ml, model, device):
    individuals = []
    _, peaks = peak_detect(spectrogram, threshold)
    for p in peaks:
        pos = p - int(patch_len / 2)
        if (pos >= 0 and len(spectrogram) >= pos+patch_len):
            ind = spectrogram[pos:pos+patch_len]
            if resize is not None:
                ind = cv2.resize(ind, dsize=resize, interpolation=cv2.INTER_NEAREST)
            if ml:
                tensor = torch.Tensor(np.expand_dims(np.expand_dims(ind, axis=0), axis=0)).to(device)
                pred = model(tensor)
                if torch.argmax(pred) == 1:
                    individuals.append(ind)
            else:
                individuals.append(ind)
    return individuals

def prepareSet(prepared_set, labels, patch_len, scale_factor, resize, one_hot, threshold, ml, model, device):
    X_ind = []
    Y_ind = []

    for species in tqdm(list(labels)):
        S_db = np.asarray(prepared_set.get(species))
        new_size = (int(S_db.shape[1] * scale_factor), int(S_db.shape[0] * scale_factor))
        S_db = cv2.resize(S_db, dsize=new_size, interpolation=cv2.INTER_NEAREST)
        label = to_categorical(labels[species], num_classes=len(labels)) if one_hot else labels[species]

        ind = getIndividuals(S_db, patch_len, resize, threshold, ml, model, device)
        X_ind.extend(ind)
        Y_ind.extend([label] * len(ind))
    
    X_ind, Y_ind = shuffle(X_ind, Y_ind, random_state=42)
    return np.asarray(X_ind), np.asarray(Y_ind)

def prepare(file, labels, patch_len, scale_factor=1.0, resize=None, one_hot=False, threshold=6, ml=False):
    prepared_hf = h5py.File(file, 'r')
    
    model = torch.jit.load('call_nocall.pt')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)

    X_train, Y_train = prepareSet(prepared_hf.require_group("train"), labels, patch_len, scale_factor,
                                  resize, one_hot, threshold, ml, model, device)
    X_test, Y_test = prepareSet(prepared_hf.require_group("test"), labels, patch_len, scale_factor,
                                resize, one_hot, threshold, ml, model, device)
    X_val, Y_val = prepareSet(prepared_hf.require_group("val"), labels, patch_len, scale_factor,
                              resize, one_hot, threshold, ml, model, device)
    return X_train, Y_train, X_test, Y_test, X_val, Y_val