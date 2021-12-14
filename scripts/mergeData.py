import pandas as pd
from sklearn.model_selection import train_test_split
import h5py

df = pd.read_csv('../data.csv')

classes = {
  "Rhinolophus ferrumequinum": [],
  "Rhinolophus hipposideros": [],
  "Myotis daubentonii": [],
  "Myotis brandtii": [],
  "Myotis mystacinus": [],
  "Myotis emarginatus": [],
  "Myotis nattereri": [],
  "Myotis bechsteinii": [],
  "Myotis myotis": [],
  "Nyctalus noctula": [],
  "Nyctalus leisleri": [],
  "Nyctalus lasiopterus": [],
  "Pipistrellus pipistrellus": [],
  "Pipistrellus pygmaeus": [],
  "Pipistrellus nathusii": [],
  "Pipistrellus kuhlii": [],
  "Hypsugo savii": [],
  "Vespertilio murinus": [],
  "Eptesicus serotinus": [],
  "Eptesicus nilssonii": [],
  "Plecotus auritus": [],
  "Plecotus austriacus": [],
  "Barbastella barbastellus": [],
  "Tadarida teniotis": [],
  "Miniopterus schreibersii": [],
  "Myotis capaccinii": [],
  "Myotis dasycneme": [],
  "Pipistrellus maderensis": [],
  "Rhinolophus blasii": []
}

labels = {
  "Rhinolophus ferrumequinum": 0,
  "Rhinolophus hipposideros": 1,
  "Myotis daubentonii": 2,
  "Myotis brandtii": 3,
  "Myotis mystacinus": 4,
  "Myotis emarginatus": 5,
  "Myotis nattereri": 6,
  "Myotis bechsteinii": 7,
  "Myotis myotis": 8,
  "Nyctalus noctula": 9,
  "Nyctalus leisleri": 10,
  "Nyctalus lasiopterus": 11,
  "Pipistrellus pipistrellus": 12,
  "Pipistrellus pygmaeus": 13,
  "Pipistrellus nathusii": 14,
  "Pipistrellus kuhlii": 15,
  "Hypsugo savii": 16,
  "Vespertilio murinus": 17,
  "Eptesicus serotinus": 18,
  "Eptesicus nilssonii": 19,
  "Plecotus auritus": 20,
  "Plecotus austriacus": 21,
  "Barbastella barbastellus": 22,
  "Tadarida teniotis": 23,
  "Miniopterus schreibersii": 24,
  "Myotis capaccinii": 25,
  "Myotis dasycneme": 26,
  "Pipistrellus maderensis": 27,
  "Rhinolophus blasii": 28
}

for index, row in df.iterrows():
  classes[row["species"]].append(row["filename"])

print("sorted!")

import librosa
import numpy as np
import soundfile as sf
import os.path
from keras.utils.np_utils import to_categorical
from scipy import signal
import cv2
from tqdm import tqdm

hop_length = 512
sample_rate = 22050
frame_rate = sample_rate / hop_length
frequency_bins = int(1025 / 10)

window_size = int(frame_rate / 2) # 500ms
overlap = int(window_size / 2) # 250ms

sequence_length = 15  # = 3.35 seconds (with overlap)
sequence_overlap = int(sequence_length / 5)

# results in shape X: (n, 11, 21, 102), Y: (n, 11)

hf = h5py.File('data.h5', 'w')

X_train = []
Y_train = []

X_test = []
Y_test = []

X_val = []
Y_val = []

b, a = signal.butter(10, 15000 / 120000, 'highpass')

def denoise(x):
  return np.abs(x - x.mean())

def slideWindow(a, size, step):
  b = []
  i = 0
  pos = 0
  while pos + size < len(a):
    pos = int(i  * step)
    b.append(a[pos : pos + size])
    i+=1
  return b

def prepare(filename):
    y, _ = librosa.load("../Chiroptera/" + filename + '.wav', sample_rate)
    filtered = signal.lfilter(b, a, y)
    D = librosa.stft(filtered, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)  #spectrogram
    S_db = np.apply_along_axis(denoise, axis=1, arr=S_db)
    new_size = (frequency_bins, len(S_db[0]))
    spectrogram = cv2.resize(S_db.transpose(), dsize=new_size, interpolation=cv2.INTER_NEAREST)
    #tiles = slideWindow(spectrogram, size=window_size, step=overlap)[:-1]
    #sequences = slideWindow(tiles, size=sequence_length, step=sequence_overlap)[:-1]
    return spectrogram

def mergeClass(name):
  files = []
  for filename in tqdm(classes[name]):
    files.append(prepare(filename))
  
  label = to_categorical(labels[name], num_classes=len(labels)) # one hot
  
  if len(files) >= 7:
    x_train, x_test, _, _ = train_test_split(files, np.zeros(len(files)), test_size=0.25, random_state=42)
    x_train, x_val, _, _ = train_test_split(x_train, np.zeros(len(x_train)), test_size=0.2, random_state=42)
    
    #x_train_c = np.concatenate(x_train)
    #x_test_c = np.concatenate(x_test)
    #x_val_c = np.concatenate(x_val)

    X_train.extend(x_train)
    Y_train.extend([label] * len(x_train))
    X_test.extend(x_test)
    Y_test.extend([label] * len(x_test))
    X_val.extend(x_val)
    Y_val.extend([label] * len(x_val))

for classname in tqdm(list(classes)):
  mergeClass(classname)

hf.create_dataset('X_train', data=X_train)
hf.create_dataset('Y_train', data=Y_train)
hf.create_dataset('X_test', data=X_test)
hf.create_dataset('Y_test', data=Y_test)
hf.create_dataset('X_val', data=X_val)
hf.create_dataset('Y_val', data=Y_val)
hf.close()