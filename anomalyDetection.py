
'''
    This file will use the One Class SVM to classify results into regular songs and anomaly songs
'''
'''
    Data fields of a given entry
    ['analysis_sample_rate', 'audio_md5', 'danceability', 'duration', 'end_of_fade_in', 
    'energy', 'idx_bars_confidence', 'idx_bars_start', 'idx_beats_confidence', 'idx_beats_start', 
    'idx_sections_confidence', 'idx_sections_start', 'idx_segments_confidence', 
    'idx_segments_loudness_max', 'idx_segments_loudness_max_time', 'idx_segments_loudness_start', 
    'idx_segments_pitches', 'idx_segments_start', 'idx_segments_timbre', 'idx_tatums_confidence', 
    'idx_tatums_start', 'key', 'key_confidence', 'loudness', 'mode', 'mode_confidence', 
    'start_of_fade_out', 'tempo', 'time_signature', 'time_signature_confidence', 'track_id']    
''' 
from pickle import load, dump
from numpy import asarray
from numpy.random import rand
from os.path import exists
import sqlite3
from sklearn.svm import OneClassSVM 
from random import random, choice, randrange
import pickle
from pandas import read_csv
from collections import Counter
#from numpy import asarray, save, load

class AnomalyDetection:
    def __init__(self, song_features, listen_history):
        self.song_features = song_features
        self.history = listen_history
        self.models = {}
        self.num_user = 0
        self.completed = 0
        self.user_histories = {}
    
    def getSongData(self, songs):
        data = []
        for i in songs:
            try:
                row = self.song_features.loc[i]
                data.append(row)
            except KeyError:
                print('Skipped')
        return data

    def getUserData(self, userID, data):
        self.user_histories[userID] = data
        songCounts = Counter(data)
        topSongs = sorted(songCounts.keys(), key=lambda x: songCounts[x], reverse=True)[:100]
        # songIDs = list(map(self.mapFunction, topSongs))
        song_data = self.getSongData(topSongs)
        model = OneClassSVM(gamma='auto').fit(song_data)
        self.models[userID] = model
        self.completed += 1
        print('Completed models for {} users'.format(self.completed), end='\r')

    def buildModels(self, svm_pickle_name, history_pickle_name):
        userName = ""
        data = []
        for user_id, lineData in self.history.itertuples():
            if userName == user_id:
                data.append(lineData)
            else:
                if userName != "":
                    self.getUserData(userName, data)
                data = [lineData]
                userName = user_id
        if svm_pickle_name:
            pickle.dump(self.models, open(svm_pickle_name, 'wb'))
        if history_pickle_name:
            pickle.dump(self.user_histories, open(history_pickle_name, 'wb'))



if __name__ == '__main__':
    song_features = read_csv("musicbrainz-data/song_features.csv", index_col='track_id')
    part1 = read_csv("musicbrainz-data/userid-trackid-1.csv", index_col='user_id')
    part2 = read_csv("musicbrainz-data/userid-trackid-2.csv", index_col='user_id')
    listen_hist = part1.append(part2)

    detector = AnomalyDetection(song_features, listen_hist)
    detector.buildModels('user_models.p', 'user_histories.p')