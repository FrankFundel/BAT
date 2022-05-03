# BAT

BAT - Bio Acoustic Transformer

# Datasets

Download the unprepared dataset from [Museum für Naturkunde Berlin, Reinald Skiba](www.tierstimmenarchiv.de/download/Chiroptera.zip) (if not available ask Dr. Karl-Heinz Frommolt for permission)
Unzip the file, move alle .WAV-files to a root folder called _data_. A .CSV file with additional information already exists called _data.csv_.

- _datasets/prepare_data.py_: FFT -> filter -> denoise -> train/test/val split -> merge spectrograms -> _prepared.h5_ (very slow, ~15GB)
- _datasets/prepare_individuals.ipynb_: Interactive notebook for extracting individuals.
- _datasets/prepare_sequences.ipynb_: Interactive notebook for extracting sequences.
- _datasets/prepare_individuals.py_: Script for extracting individuals.
- _datasets/prepare_sequences.py_: Script for extracting sequences.
- _datasets/call_nocall.ipynb_: Interactive notebook for annotating individuals into call/no-call -> _call_nocall.annotation_


# Metadata

Geographical and weather data of each recording.

- _geoweather/collect.py_: Collects both geographical data and weather data and saves it to a .JSON file
- _geoweather/geoTags.py_: Collects geographical data from OpenStreetMap of a location
- _geoweather/weatherData.py_: Collects weather data from meteostat of a location and time


# Models
- _models_/baseline.ipynb_: ResNet-50 for species classification using individual calls -> [_baseline.pt_](https://drive.google.com/file/d/1XDiJwc8qToNIGQ_hQzJFNvIWeih2-iFE/view?usp=sharing), _baseline_cf.png_
- _models_/call_nocall.ipynb_: ResNet-18 for call/no-call classification using individual calls -> [_call_nocall.pt_](https://drive.google.com/file/d/19J7m7xPEoUOjANC7bETB6RQW9vIkQskQ/view?usp=sharing)


# Pretrained

# Publication

# Implementation
