import csv
import json
from datetime import datetime
from time import mktime
import numpy as np

from geoTags import getGeoTags
from weatherData import getWeatherData

system = {
  "Rhinolophus ferrumequinum": "0",
  "Rhinolophus hipposideros": "1",
  "Myotis daubentonii": "2",
  "Myotis brandtii": "3",
  "Myotis mystacinus": "4",
  "Myotis emarginatus": "5",
  "Myotis nattereri": "6",
  "Myotis bechsteinii": "7",
  "Myotis myotis": "8",
  "Nyctalus noctula": "9",
  "Nyctalus leisleri": "10",
  "Nyctalus lasiopterus": "11",
  "Pipistrellus pipistrellus": "12",
  "Pipistrellus pygmaeus": "13",
  "Pipistrellus nathusii": "14",
  "Pipistrellus kuhlii": "15",
  "Hypsugo savii": "16",
  "Vespertilio murinus": "17",
  "Eptesicus serotinus": "18",
  "Eptesicus nilssonii": "19",
  "Plecotus auritus": "20",
  "Plecotus austriacus": "21",
  "Barbastella barbastellus": "22",
  "Tadarida teniotis": "23",
  "Miniopterus schreibersii": "24",
  "Myotis capaccinii": "25",
  "Myotis dasycneme": "26",
  "Myotis emarginatus": "27",
  "Myotis mystacinus": "28",
  "Pipistrellus maderensis": "29",
  "Rhinolophus blasii": "30"
}

#filename
#species
#recording_date
#recording_time
#description
#duration
#latitude
#longitude

'''Ultraschallaufnahmen im Zeitdehnverfahren 1:10'''
'''Pettersson D980 Time expansion. copy to Sony WM D6C'''
'''Sample Rate: 96000, Bit depth: 24'''

data = []

with open('..\data.csv', newline='') as csvfile:
  print("test")
  reader = csv.reader(csvfile, delimiter=',')

  header = next(reader)
  for i in range(len(header)):
    print(str(i) + ": " + header[i])
  
  id = 0
  for row in reader:
    if(row[10] != "0000-00-00" and row[7] != "0" and row[8] != "0"):
      latitude = float(row[7])
      longitude = float(row[8])
      geoTags = []
      if (latitude and longitude):
        geoTags = getGeoTags(latitude, longitude)

      date = row[10]
      time = "20:00:00" # row[11]
      dt = datetime.strptime(date + " " + time, '%Y-%m-%d %H:%M:%S')
      if(dt.year >= 1999):
        recordingDateTime = mktime(dt.timetuple())
        weather = getWeatherData(latitude, longitude, dt)

      obj = {
        "id": id,
        "filename": row[34],
        "species": int(system[row[0]]),

        "duration": row[33],
        "recordingDateTime": recordingDateTime,

        "location": {
          "latitude": latitude,
          "longitude": longitude,
        },
        "geoTags": geoTags,
        "weatherTags": weather,
        "description": row[17]
      }
      data.append(obj)
      print(obj)
    id+=1

with open('data.json', 'w') as outfile:
  json.dump(data, outfile)