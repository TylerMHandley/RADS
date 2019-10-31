
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
import hdf5_getters
from sklearn.svm import OneClassSVM 
from multiprocessing import Process, Value, Manager, Lock
from random import random, choice, randrange
import pickle
#from numpy import asarray, save, load

class AnomalyDetection:
    def __init__(self, filename):
        self.filename = filename
        self.models = {}
        self.num_user = 0
        self.completed = Value('i', 0)
    
    def generateRandom(self):
        val1 = randrange(-30, 30)/100 * -7.75
        val2 = randrange(-30, 30)/100 * 113.359
        random_song = [ random(), choice([1,2,3]), -7.75+val1, 113.359+val2, random(), random(), random()]
        return random_song

    def getSongData(self, songs):
        h5 = hdf5_getters\
        .open_h5_file_read('MillionSongSubset/AdditionalFiles/subset_msd_summary_file.h5')
        data = []
        for i in songs:
            if i[0] == 'S':
                data.append(self.generateRandom())
            else:
                #print(row['idx_segments_pitches'])
                for index in h5.root.analysis.songs.get_where_list('track_id=={}'.format(str.encode(i))):
                    row = h5.root.analysis.songs[index]
                    songInfo = [row['energy'], row['mode'], row['loudness'], row['tempo'], row['danceability'], 
                    row['idx_segments_pitches'], row['idx_segments_timbre']]
                    # segmentPitch = asarray(row['idx_segments_pitches']).flatten()
                    # segmentTimbre = asarray(row['idx_segments_timbre']).flatten()
                    # songInfo.extend(segmentPitch)
                    # songInfo.extend(segmentTimbre)
                    data.append(songInfo)
                # rows = h5.root.analysis.songs.get_where_list('track_id=={}'.format(str.encode(track_id)))
                # row = h5.root.analysis.songs[rows[0]]
                # songInfo = [row['energy'], row['mode'], row['loudness'], row['tempo'], row['danceability'],
                # row['idx_segments_pitches'], row['idx_segments_timbre']]
                # data.append(songInfo)
        return data
    
    def mapFunction(self, x):
        if x[1] > 0:
            return x[0]

    def getUserData(self, userID, data, model_dict, lock):
        topSongs = sorted(data, key=lambda x: x[1], reverse=True)
        songIDs = list(map(self.mapFunction, topSongs))
        song_data = self.getSongData(songIDs)
        model = OneClassSVM(gamma='auto').fit(song_data)
        model_dict[userID] = model
        with lock:
            self.completed.value +=1
            print('Completed models for {} users'.format(self.completed.value), end='\r')

    def buildModels(self, pickleName):
        with open(self.filename, 'r') as read_file:
            userName = ''
            data = []
            pool_num = 0
            pool = []
            lock = Lock()
            with Manager() as mnger:
                shared_dict = mnger.dict()
                for line in read_file:
                    if self.completed.value > 1000:
                        break
                    lineData = line.split('\t')
                    if userName == lineData[0]:
                        data.append((lineData[1], int(lineData[2])))
                    else:
                        if userName != '':
                            self.num_user+=1
                            p = Process(target=self.getUserData, args=(userName, data, shared_dict, lock))
                            pool.append(p)
                            pool_num+=1
                            p.start()
                        if pool_num >= 10:
                            for proc in pool:
                                proc.join()
                            pool = []
                            pool_num = 0
                        userName = lineData[0]
                        data = [(lineData[1], int(lineData[2]))]
                if pool_num > 0:
                    for i in pool:
                        i.join()
                self.models = shared_dict
        pickle.dump(self.models, open(pickleName, 'wb'))        



if __name__ == '__main__':
    detector = AnomalyDetection('200000_user_data.txt')
    detector.buildModels('user_models.p')