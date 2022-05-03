import sys
import io
import json
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS, cross_origin
sys.path.append('../datasets/')
sys.path.append('../models/')

from prepare_data import prepareData
from prepare_sequences import getSequences, germanBats
from prepare_individuals import getIndividuals

import soundfile as sf
import torch
import librosa
import numpy as np
import cv2

app = Flask(__name__)
cors = CORS(app)


sample_rate = 22050          # recordings are in 96 kHz, 24 bit depth, 1:10 TE (mic sr 960 kHz), 22050 Hz = 44100 Hz TE


def prepare(audio_bytes, expanded):
    # load bytes as signal
    tmp = io.BytesIO(audio_bytes)
    data, sr = sf.read(tmp, dtype='float32')
    data = data.T
    if expanded == "false":
        y = librosa.resample(data, sr, sample_rate*10)
    else:
        y = librosa.resample(data, sr, sample_rate)
    S_db = prepareData(y) # filter, spectrogram, denoise
    return S_db


def get_prediction(audio_bytes, selected_model, expanded):
    # select model
    if selected_model == "BAT-1: 18 european bats":
        classes = germanBats
        patch_len = 44               # = 250ms ~ 25ms
        patch_skip = 22              # = 150ms ~ 15ms
        seq_len = 60                 # = 500ms with ~ 5 calls
        seq_skip = 15
        resize = (64, 44)
        model = torch.jit.load('../models/bat_1.pt')
        
    elif selected_model == "ResNet-50: 18 european bats":
        classes = germanBats
        patch_len = 44               # = 250ms ~ 25ms
        patch_skip = 22              # = 150ms ~ 15ms
        resize = None
        model = torch.jit.load('../models/baseline.pt')
        
    else:
        return np.zeros(len(classes)).tolist(), classes
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)
    model.eval()
    
    S_db = prepare(audio_bytes, expanded)
        
    if selected_model == "BAT-1: 18 european bats":
        sequences = np.asarray(getSequences(S_db, patch_len, patch_skip, seq_len, seq_skip, resize))
        tensor = torch.Tensor(sequences).to(device)
        outputs = model(tensor)
        outputs = torch.nn.functional.softmax(outputs.mean(dim=0), dim=0)
        return outputs.tolist(), classes
    
    elif selected_model == "ResNet-50: 18 european bats":
        inds = np.asarray(getIndividuals(S_db, patch_len))
        calls = []
        call_nocall_model = torch.jit.load('../models/call_nocall.pt').to(device)
        call_nocall_model.eval()
        for x in inds:
            tensor = torch.Tensor(np.expand_dims(np.expand_dims(x, axis=0), axis=0)).to(device)
            out = call_nocall_model(tensor)
            if torch.argmax(out) == 1:
                calls.append(x)
        calls = np.asarray(calls)
        
        if calls.shape[0] > 0:
            tensor = torch.Tensor(np.expand_dims(calls[:64], axis=1)).to(device)
            outputs = model(tensor)
            outputs = torch.nn.functional.softmax(outputs.mean(dim=0), dim=0)
            return outputs.tolist(), classes
    
    return np.zeros(len(classes)).tolist(), classes

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        selected_model = request.form['model']
        file = request.files['file']
        expanded = request.form['expanded']
        
        audio_bytes = file.read()
        prediction, classes = get_prediction(audio_bytes, selected_model, expanded)
        return jsonify({'prediction': prediction, 'classes': list(classes)})

    
@app.route('/play', methods=['POST'])
@cross_origin()
def play():
    if request.method == 'POST':
        data = json.loads(request.form['data'])
        S_db = np.asarray(data)
        S_db = cv2.resize(S_db, dsize=(44, 257), interpolation=cv2.INTER_NEAREST)
        silence = np.zeros((176, 257))
        extended = np.concatenate([silence, S_db.transpose(), silence]).transpose()
        mapped = (extended - 1)*80
        S = librosa.db_to_amplitude(mapped, ref=80)
        y = librosa.griffinlim(S)
        sf.write('out.wav', y, sample_rate, 'PCM_24')
        return send_file("out.wav")
    

# A welcome message to test our server
@app.route('/')
def index():
    return "<h1>Welcome to our server !!</h1>"
    

if __name__ == '__main__':
    app.run(threaded=True, port=5000)