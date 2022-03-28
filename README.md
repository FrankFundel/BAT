# BAT

BAT - Bio Acoustic Transformer

# Datasets

- _load.py_: Downloads data and unzips
- _merge.py_: FFT -> filter -> denoise -> train/test/val split -> merge -> h5
- _sequence.py_: sequencing -> h5, to wave -> sequencing -> h5
- _separate.py_: separate -> h5, to wave -> separate -> h5

# GeoWeather

Geographical and weather data of each recording.

- _collect.py_: Collects both geographical data and weather data and saves it to a .JSON file
- _geoTags.py_: Collects geographical data from OpenStreetMap of a location
- _weatherData.py_: Collects weather data from meteostat of a location and time

# Training

# Pretrained

# Publication

# Implementation
