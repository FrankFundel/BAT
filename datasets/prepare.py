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

classes23 = {
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
  "Rhinolophus hipposideros": 12,
  "Myotis brandtii": 13,
  "Myotis mystacinus": 14,
  "Myotis emarginatus": 15,
  "Myotis myotis": 16,
  "Pipistrellus pygmaeus": 17,
  "Hypsugo savii": 18,
  "Eptesicus nilssonii": 19,
  "Tadarida teniotis": 20,
  "Myotis capaccinii": 21,
  "Pipistrellus maderensis": 22,
  "Rhinolophus blasii": 23
}

labels = classes13

merged_hf = h5py.File('merged.h5', 'r')

sample_rate = 22050                         # recordings are in 96 kHz, 24 bit depth, 1:10 TE (mic sr 960 kHz), 22050 Hz = 44100 Hz TE
n_fft = 512                                 # 23 ms * 22050 Hz
frame_rate = sample_rate / (n_fft // 4)	    # 22050 / 128 = 256

patch_len = 43           		    # = 250ms ~ 25ms
patch_skip = 25			            # = 150ms ~ 15ms
seq_length = 30			    	    # = 500ms with ~ 5 calls
seq_skip = 15

scale_factor = 1.0

def slideWindow(a, size, step):
  b = []
  i = 0
  pos = 0
  while pos + size < len(a):
    pos = int(i * step)
    b.append(a[pos : pos + size])
    i+=1
  return b

def getSequences(spectrogram):
  tiles = slideWindow(spectrogram, size=path_len, step=patch_skip)[:-1] # last one is not full
  sequences = slideWindow(tiles, size=seq_len, step=seq_skip)[:-1] # last one is not full
  return sequences

def getIndividuals():
  # return sequences
  return


  for set in ["train", "test", "val"]:
    merged_set = merged_hf.require_group(set)

    seq_hf = h5py.File('sequences_s.h5', 'a')
    #ind_hf = h5py.File('individuals_s.h5', 'a')

    X_seq = []
    Y_seq = []
    #X_ind = []
    #Y_ind = []
    print("Preparing " + set + " dataset.")

    for species in tqdm(list(labels)):
      S_db = merged_set.get(species)
      new_size = S_db.shape * scale_factor
      S_db = cv2.resize(np.float32(S_db), dsize=new_size, interpolation=cv2.INTER_NEAREST)
      label = to_categorical(labels[species], num_classes=len(labels)) # one hot encoding
      seq = getSequences(S_db)
      print(S_db.shape)
      X_seq.extend(seq)
      Y_seq.extend([label] * len(seq))

      #ind = getIndividuals(S_db)
      #X_ind.extend(ind)
      #Y_ind.extend([label] * len(ind))

    print("Sequences:", len(X_seq))
    #print("Individuals:", len(X_ind))

    print("Shuffling...")
    X_seq, Y_seq = shuffle(X_seq, Y_seq, random_state=42)

    print("Writing...")
    seq_hf.create_dataset('X_' + set, data=X_seq)
    seq_hf.create_dataset('Y_' + set, data=Y_seq)
    seq_hf.close()
    del X_seq
    del Y_seq
    
    #X_ind, Y_ind = shuffle(X_ind, Y_ind, random_state=42)
    #ind_hf.create_dataset('X_' + set, data=X_ind)
    #ind_hf.create_dataset('Y_' + set, data=Y_ind)
    #ind_hf.close()