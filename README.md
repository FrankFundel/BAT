# BAT

BAT - Bio Acoustic Transformer

# Datasets

Download the unprepared dataset from [Museum für Naturkunde Berlin, Reinald Skiba](www.tierstimmenarchiv.de/download/Chiroptera.zip) (sometimes not available)
Unzip the file, move alle .WAV-files to a root folder called _data_. A .CSV file with additional data already exists called _data.csv_.

- _datasets/prepared.py_: FFT -> filter -> denoise -> train/test/val split -> merge recordings -> _prepared.h5_ (very slow, ~15GB)
- _datasets/prepare_sequences.ipynb_: Interactive notebook for sequencing (fast, in memory)
- _datasets/prepare_individuals.ipynb_: Interactive notebook for individuals (fast, in memory)

# Metadata

Geographical and weather data of each recording.

- _geoweather/collect.py_: Collects both geographical data and weather data and saves it to a .JSON file
- _geoweather/geoTags.py_: Collects geographical data from OpenStreetMap of a location
- _geoweather/weatherData.py_: Collects weather data from meteostat of a location and time

# Training

# Pretrained

# Publication

# Implementation
