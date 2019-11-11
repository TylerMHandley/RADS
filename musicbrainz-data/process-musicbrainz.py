from pandas import read_csv, DataFrame
from requests import get
from numpy import array

def get_response(tracks: array):
    return get('https://acousticbrainz.org/api/v1/high-level', {'recording_ids': ';'.join(tracks)})

cols = ['user_id', 'timestamp', 'artist_id', 'artist_name', 'track_id', 'track_name']
print("Loading data from file. This takes a bit...")
data = read_csv('userid-timestamp-artid-artname-traid-traname.tsv', sep='\t', names=cols)

# Get rid of rows that are missing fields
print("Finding unique tracks")
data = data.dropna()
unique_tracks = data['track_id'].unique()
num_unique = len(unique_tracks)
print('Total of {} tracks'.format(num_unique))
track_data = []

print('Making requests to Acousticbrainz API...')
try:
    for i in range(0, num_unique, 25):
        response = get_response(unique_tracks[i:i+25])
        if response.status_code != 200:
            print("stopped receiving responses at track_id {}, i={}".format(unique_tracks[i], i))
            break
        for track_id in response.json():
            try:
                features = response.json()[track_id]['0']['highlevel']
                # Get all features from responses
                danceability = features['danceability']['all']['danceable']
                gender = features['gender']['all']['female']
                acoustic = features['mood_acoustic']['all']['acoustic']
                aggressive = features['mood_aggressive']['all']['aggressive']
                electronic = features['mood_electronic']['all']['electronic']
                happy = features['mood_happy']['all']['happy']
                party = features['mood_party']['all']['party']
                relaxed = features['mood_relaxed']['all']['relaxed']
                sad = features['mood_sad']['all']['sad']
                mirex_c1 = features['moods_mirex']['all']['Cluster1']
                mirex_c2 = features['moods_mirex']['all']['Cluster2']
                mirex_c3 = features['moods_mirex']['all']['Cluster3']
                mirex_c4 = features['moods_mirex']['all']['Cluster4']
                mirex_c5 = features['moods_mirex']['all']['Cluster5']
                timbre = features['timbre']['all']['bright']
                tonal = features['tonal_atonal']['all']['tonal']
                instrumental = features['voice_instrumental']['all']['instrumental']
                
                # Append data to list
                track_data.append([track_id, danceability, gender, acoustic, aggressive,\
                    electronic, happy, party, relaxed, sad, mirex_c1, mirex_c2, mirex_c3,\
                    mirex_c4, mirex_c5, timbre, tonal, instrumental])
            except KeyError as e:
                print('track_id {} missing feature {}'.format(track_id, e.args[0]))
        print('Finished {}-{}'.format(i, i+25))
except KeyboardInterrupt:
    print('User ended process early, saving results')
except OSError:
    print('Network is down, saving results')

feat_cols = ['track_id', 'danceability', 'gender', 'acoustic', 'aggressive', 'electronic',\
    'happy', 'party', 'relaxed', 'sad', 'mirex_c1', 'mirex_c2', 'mirex_c3', 'mirex_c4',\
    'mirex_c5', 'timbre', 'tonal', 'instrumental']

# Save features to file
song_feats = DataFrame(track_data, columns=feat_cols)
song_feats.to_csv('song_features2.csv', index=False)
