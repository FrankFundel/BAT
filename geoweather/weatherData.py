from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from meteostat import Point, Hourly

def getWeatherData(latitude, longitude, datetime):
  center = Point(latitude, longitude)
  data = Hourly(center, datetime, datetime + timedelta(hours=1))
  data = data.fetch()
  out = {}
  if (not data.empty and data.size > 0): 
    row = data.iloc[0]
    out = {
      "temp": row[0],
      "dewpoint": row[1],
      "humidity": row[2],
      "rain": row[3],
      "windspeed": row[6]
    }
  return out
  
# test
# print(getWeatherData(52.36, 13.67, datetime.fromtimestamp(932754827)))
# # Ziegenhals (bei Wernsdorf), Fri Jul 23 1999 20:27:22