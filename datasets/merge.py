import pandas as pd
from sklearn.model_selection import train_test_split
import librosa
import numpy as np
from scipy import signal
from tqdm import tqdm
import h5py

df = pd.read_csv('../data.csv')
hf = h5py.File('merged.h5', 'a')  # will be ~13GB
train_set = hf.require_group("train")
test_set = hf.require_group("test")
val_set = hf.require_group("val")

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

for index, row in df.iterrows():
  classes[row["species"]].append(row["filename"])

print("sorted!")

hop_length = 512

b, a = signal.butter(10, 15000 / 120000, 'highpass')       # 1th order butterworth high-pass filter with cut-off frequency of 15,000 kHz

def denoise(x):
  return np.abs(x - x.mean())

def mergeClass(name):
  signals = []
  for filename in tqdm(classes[name]):
    y, _ = librosa.load("../data/" + filename + '.wav')
    filtered = signal.lfilter(b, a, y)                      # filter
    D = librosa.stft(filtered, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)   # spectrogram
    S_db = np.apply_along_axis(denoise, axis=1, arr=S_db)   # denoise
    signals.append(np.transpose(S_db))
  
  if len(signals) >= 7:
    X_train, X_test, _, _ = train_test_split(signals, np.zeros(len(signals)), test_size=0.25, random_state=42)
    X_train, X_val, _, _ = train_test_split(X_train, np.zeros(len(X_train)), test_size=0.2, random_state=42)
    train = np.concatenate(X_train)
    test = np.concatenate(X_test)
    val = np.concatenate(X_val)
    train_set.create_dataset(name, data=train)
    test_set.create_dataset(name, data=test)
    val_set.create_dataset(name, data=val)

for classname in list(classes):
  if not classname in train_set or not classname in test_set or not classname in val_set:
    mergeClass(classname)
    print(classname + " merged!")

hf.close()