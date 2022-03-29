# open merged.h5
# read arguments (s, w, sm)

# create individuals_s.h5
# create sequences_s.h5
# for each set (train, test, val):
#   for each classname:
#     getIndividuals(S_db)
#     getSequences(S_db)

# create individuals_w.h5
# create sequences_w.h5
# for each set (train, test, val):
#   for each classname:
#     S = librosa.db_to_amplitude(S_db, ref=np.max)
#     y = librosa.griffinlim(S, hop_length=hop_length)
#     getIndividuals(y)
#     getSequences(y)