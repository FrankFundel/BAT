# BAT

BAT - Bio Acoustic Transformer

# Datasets

Download the unprepared dataset from [Museum für Naturkunde Berlin, Reinald Skiba](www.tierstimmenarchiv.de/download/Chiroptera.zip) (sometimes not available)
Unzip the file, move alle .WAV-files to a root folder called _data_. A .CSV file with additional data already exists called _data.csv_.

- _datasets/merge.py_: FFT -> filter -> denoise -> train/test/val split -> merge recordings -> _merged.h5_
- _datasets/sequence.py_: sequencing -> h5, to wave -> sequencing -> h5
- _datasets/separate.py_: separate -> h5, to wave -> separating -> h5

# Metadata

Geographical and weather data of each recording.

- _geoweather/collect.py_: Collects both geographical data and weather data and saves it to a .JSON file
- _geoweather/geoTags.py_: Collects geographical data from OpenStreetMap of a location
- _geoweather/weatherData.py_: Collects weather data from meteostat of a location and time

# Training

# Pretrained

# Publication

# Implementation
