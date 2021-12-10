import overpy

api = overpy.Overpass()

def getGeoTags(latitude, longitude):
  center = str(latitude) + ", " + str(longitude)
  result = api.query("""
      [out:json];
      way(around:500.0, """ + center + """)->.a;
      (
        way.a[amenity];
        way.a[natural];
        way.a[landuse];
      );
      (._;>;);
      out;
      """)

  tags = []

  for way in result.ways:
    tags.append(way.tags.get("natural"))
    tags.append(way.tags.get("landuse"))

  #categories:
  buildings = ["commercial", "education", "industrial", "residential", "retail", "garages", "greenhouse_horticulture", "military", "religious"]
  lifeless = ["construction", "depot", "landfill", "brownfield", "quarry", "bare_rock", "shingle", "sandbeach"]
  lawn = ["heath", "moor", "grassland", "fell", "tundra", "mud", "wetland", "allotments", "farmland", "farmyard", "meadow", "orchard", "vineyard", "cemetery", "grass", "greenfield", "plant_nursery", "recreation_ground", "village_green"]
  forest = ["tree_row", "wood", "tree", "forest"] #leaf_type=needleleaved/broadleaved/mixed/leafless
  water = ["water", "cape", "bay", "strait", "coastline", "reef", "spring", "aquaculture", "basin", "reservoir", "salt_pond"]
  mountain = ["peak", "dune", "hill", "cliff", "rock", "stone", "cave_entrance"]

  out = {}

  if any(item in tags for item in buildings):
    out["buildings"] = True
  else:
    out["buildings"] = False

  if any(item in tags for item in lifeless):
    out["lifeless"] = True
  else:
    out["lifeless"] = False
    
  if any(item in tags for item in lawn):
    out["lawn"] = True
  else:
    out["lawn"] = False
    
  if any(item in tags for item in forest):
    out["forest"] = True
  else:
    out["forest"] = False
    
  if any(item in tags for item in water):
    out["water"] = True
  else:
    out["water"] = False
    
  if any(item in tags for item in mountain):
    out["mountain"] = True
  else:
    out["mountain"] = False

  return out

# print(getGeoTags(52.36, 13.67))