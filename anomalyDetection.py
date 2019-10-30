
'''
    This file will use the One Class SVM to classify results into regular songs and anomaly songs
'''

from pickle import load, dump
from os.path import exists
import sqlite3
import hdf5_getters
from sklearn.svm import OneClassSVM 
#from numpy import asarray, save, load

class AnomalyDetection:
    def __init__(self, filename):
        self.filename = filename
        self.models = {}
    
    def mapFunction(self, x):
        if int(x[1]) > 0:
            return x[0]

    def trainSVM(self, userID, data):
        topSongs = sorted(data, key=lambda x: x[1], reverse=True)
        songIDs = list(map(self.mapFunction, topSongs))
        trainingData = self.getSongData(songIDs)
        model = OneClassSVM().fit(trainingData)
        self.models[userID] = model    
        
    # energy, mode, loudness, tempo, segment_pitches, segments_timbre, danceability
    def getSongData(self, songs):
        h5 = hdf5_getters\
        .open_h5_file_read('MillionSongSubset/AdditionalFiles/subset_msd_summary_file.h5')
        data = []
        for i in songs:
            rowIter = h5.root.analysis.songs.where('track_id=={}'.format(str.encode(i)))
            for row in rowIter:
                songInfo = [row['energy'], row['mode'], row['loudness'], row['tempo'], row['segment_pitches'], row['segments_timbre'], row['dancebility']]
                data.append(songInfo)
                break
        return data
        
    # This looks in the sql database, which is very limited
    # def getSongData(self, songs):
    #     tm_conn = sqlite3.connect('MillionSongSubset/AdditionalFiles/subset_track_metadata.db')
    #     found_songs = []
    #     for songId in songs:
    #         query = "SELECT * FROM songs WHERE song_id=?"
    #         res = tm_conn.execute(query, (songId,))
    #         fetched_value = res.fetchall() 
    #         if len(fetched_value) > 0:
    #             print(fetched_value)
    #             found_songs.append(fetched_value[0])
    #     print(len(found_songs))

    def getAllUserData(self):
        # This function will eat all your RAM
        # The data initially has each song per user on a different line, here we
        # concatenate all those together into a list where an index  is a list where the 
        # first element is userID and the rest is tuples of songID and counts sorted in reverse order.
        # File data format is userID\tsongID\tcount
        if exists('userData.p'):
            print('Existing user file found. Loading that...')
            userInfo = load(open('userData.p', 'rb'))
            #userInfo = load('userData.npy')
            print('Loading Complete.')
            return userInfo
        userInfo = []
        previousUser = ''
        songList = []
        print('User file not found. Loading data from {}'.format(filename))
        with open(self.filename, 'r') as inputFile:
            for line in inputFile:
                current = line.split('\t')
                if current[0] != previousUser: 
                    if previousUser != '':
                        songList.sort(reverse=True, key=lambda x: x[1])
                        userInfo.append(songList)
                    previousUser = current[0]
                    songList=[previousUser, (current[1], current[2])]
                songList.append((current[1], current[2]))
            userInfo.append(songList)
        print('Loading Complete.\nDumping data to userData.p')
        dump(userInfo, open('userData.p', 'wb'))
        #userInfo = asarray(userInfo)
        #save('userData', userInfo)
        return userInfo
    def processUserData(self):
        with open(self.filename, 'r') as file:
            userName = ''
            data = []
            for line in file:
                lineData = line.split('\t')
                if userName == lineData[0]:
                    data.append((lineData[1], lineData[2]))
                else:
                    if userName != '':
                        self.trainSVM(userName, data)
                    userName = lineData[0]
                    data = [(lineData[1], lineData[2])]



if __name__ == '__main__':
    detector = AnomalyDetection('train_triplets.txt')
    detector.processUserData()