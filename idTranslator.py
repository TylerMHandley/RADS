
'''
    This file will use the One Class SVM to classify results into regular songs and anomaly songs
'''

from pickle import load, dump
from os.path import exists
import sqlite3
import hdf5_getters
# from sklearn.svm import OneClassSVM 
# from multiprocessing import Process, Value, Manager, Lock
#from numpy import asarray, save, load
tm_conn = sqlite3.connect('MillionSongSubset/AdditionalFiles/subset_track_metadata.db')
class Translator:
    def __init__(self, filename):
        self.filename = filename
        self.completed = 0

    def translateUserData(self, userID, data):
        trackIDs = self.getTrackIDs(data)
        stringToWrite = ''
        for i in trackIDs:
            stringToWrite += "{}\t{}\t{}\n".format(userID, i[0], i[1])
        self.completed += 1
        print('Completed {}'.format(self.completed), end='\r')
        return stringToWrite

    def getTrackIDs(self, songs):
        trackID_pairs = []
        for i in songs:
            query = "SELECT track_id FROM songs WHERE song_id=?"
            res = tm_conn.execute(query, (i[0],))
            response_data = res.fetchall()
            track_id = i[0]
            if len(response_data) > 0:
                track_id = response_data[0][0]
            trackID_pairs.append((track_id, i[1]))
        return trackID_pairs
    
    def processUserData(self):
        with open(self.filename, 'r') as read_file:
            userName = ''
            data = []
            with open('user_data.txt', 'a+') as writeFile:
                for line in read_file:
                    lineData = line.split('\t')
                    if userName == lineData[0]:
                        data.append((lineData[1], int(lineData[2])))
                    else:
                        if userName != '':
                            writeFile.write(self.translateUserData(userName, data))
                        userName = lineData[0]
                        data = [(lineData[1], int(lineData[2]))]



if __name__ == '__main__':
    trans = Translator('train_triplets.txt')
    trans.processUserData()