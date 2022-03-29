import h5py
import sys
from tqdm import tqdm
import librosa
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle

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

merged_hf = h5py.File('merged-1025.h5', 'r')

mode = sys.argv[1]

sample_rate = 22050                         # recordings are in 96 kHz, 24 bit depth, 1:10 TE (mic sr 960 kHz), 22050 Hz = 44100 Hz TE
n_fft = 512                                 # 10 ms = 100 ms TE * sample_rate (e.g. 23 ms * 22050 Hz ~ 512)
hop_length = 512
frame_rate = sample_rate / hop_length

window_size = int(frame_rate / 2)           # /2 = 500ms ~ 50ms
overlap = int(window_size / 2)              # /2 = 250ms ~ 25ms
sequence_length = 8                         # 15 = 3.35 seconds, 30 = 7.5 seconds (with overlap)
sequence_overlap = int(sequence_length / 4)

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
  tiles = slideWindow(spectrogram, size=window_size, step=overlap)[:-1] # last one is not full
  sequences = slideWindow(tiles, size=sequence_length, step=sequence_overlap)[:-1] # last one is not full
  return sequences

def getIndividuals():
  # return sequences
  return

if (mode == 's'):
  seq_hf = h5py.File('sequences_s.h5', 'a')
  #ind_hf = h5py.File('individuals_s.h5', 'a')

  for set in ["train", "test", "val"]:
    merged_set = merged_hf.require_group(set)
    X_seq = []
    Y_seq = []
    #X_ind = []
    #Y_ind = []

    for species in tqdm(list(labels)):
      S_db = merged_set.get(species)
      label = to_categorical(labels[species], num_classes=len(labels)) # one hot encoding
      seq = getSequences(S_db)
      X_seq.extend(seq)
      Y_seq.extend([label] * len(seq))

      #ind = getIndividuals(S_db)
      #X_ind.extend(ind)
      #Y_ind.extend([label] * len(ind))

    X_seq, Y_seq = shuffle(X_seq, Y_seq, random_state=42)
    seq_hf.create_dataset('X_' + set, data=X_seq)
    seq_hf.create_dataset('Y_' + set, data=Y_seq)
    #X_ind, Y_ind = shuffle(X_ind, Y_ind, random_state=42)
    #ind_hf.create_dataset('X_' + set, data=X_ind)
    #ind_hf.create_dataset('Y_' + set, data=Y_ind)
  
  seq_hf.close()
  print("Sequences:", len(X_seq))
  #ind_hf.close()
  #print("Individuals:", len(X_ind))

'''
elif (mode == 'w'):
  seq_hf = h5py.File('sequences_w.h5', 'a')
  ind_hf = h5py.File('individuals_w.h5', 'a')

  for set in ["train", "test", "val"]:
    merged_set = merged_hf.require_group(set)
    seq_set = seq_hf.require_group(set)
    ind_set = ind_hf.require_group(set)
    for species in tqdm(list(labels)):
      S_db = merged_set.require_dataset(species)
      S = librosa.db_to_amplitude(S_db, ref=np.max)
      y = librosa.griffinlim(S, n_fft=n_fft)
      seq_set.create_dataset(species, data=getSequences(y))
      ind_set.create_dataset(species, data=getIndividuals(y))

elif (mode == 'sm'):
  print("TODO")'''