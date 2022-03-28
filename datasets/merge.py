import pandas as pd
from sklearn.model_selection import train_test_split

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

for index, row in df.iterrows():
  classes[row["species"]].append(row["filename"])

print("sorted!")

import librosa
import numpy as np
import soundfile as sf
import os.path

def mergeClass(name):
  signals = []
  for filename in classes[name]:
    y, sr = librosa.load("../Chiroptera/" + filename + '.wav')
    signals.append(y)
  if len(signals) >= 7:
    X_train, X_test, _, _ = train_test_split(signals, np.zeros(len(signals)), test_size=0.25, random_state=42)
    X_train, X_val, _, _ = train_test_split(X_train, np.zeros(len(X_train)), test_size=0.2, random_state=42)
    train = np.concatenate(X_train)
    test = np.concatenate(X_test)
    val = np.concatenate(X_val)
    sf.write("../train/" + name + '.wav', train, sr, 'PCM_24')
    sf.write("../test/" + name + '.wav', test, sr, 'PCM_24')
    sf.write("../val/" + name + '.wav', val, sr, 'PCM_24')
  else:
    print("no signal for " + name)

for classname in list(classes):
  if not os.path.exists("../train/" + classname + '.wav') and not os.path.exists("../test/" + classname + '.wav') and not os.path.exists("../val/" + classname + '.wav'):
    mergeClass(classname)
    print(classname + " merged!")