from anomalyDetection import AnomalyDetection
from os.path import exists
from pickle import load, dump
from pandas import read_csv
from multiprocessing import Process, Manager


def loadData(filenames, index_column):
        data = read_csv(filenames[0], index_col=index_column)
        if len(filenames) > 1:
            for i in range(1, len(filenames)):
                extra_part = read_csv(filenames[i], index_col=index_column)
                data.append(extra_part)
        return data

class RADS:
    def __init__(self, song_files, user_files, svm_pickle_filename=None, history_pickle_filename=None):
        self.radsData = None
        self.userModels = None
        self.song_data = loadData(song_files, 'track_id')
        #self.user_history_data = loadData(user_files, 'user_id')
        self.user_history_files = user_files
        self.svm_pickle_filename = svm_pickle_filename
        self.history_pickle_filename = history_pickle_filename
        self.getSVMs()

    def getSVMs(self):
        if exists(self.svm_pickle_filename):
            self.userModels = load(open(self.svm_pickle_filename, 'rb'))
            print("Pickled State Loaded from {}".format(self.svm_pickle_filename))
        else:
            print("Pickled file not found, generating SVMs...")
            user_data = loadData(self.user_history_files, 'user_id')
            anomalyDetector = AnomalyDetection(self.song_data, user_data)
            anomalyDetector.buildModels(self.svm_pickle_filename, self.history_pickle_filename)
            self.userModels = anomalyDetector.models

    def generate(self, pickle_filename):
        if exists(pickle_filename):
            self.radsData = load(open(pickle_filename, 'rb'))
        else:
            x = 20 
            indexes = list(self.song_data.index)
            indexes = indexes[:len(indexes)//x]
            total_ids = len(indexes)
            split_val = total_ids//8
            self.radsData = dict.fromkeys(indexes)      
            with Manager() as manager:
                man_dict = manager.dict(self.radsData)
                pool = []
                count = 0
                first = 0
                second = split_val
                while(second<total_ids):
                    data = indexes[first:second]
                    #features = self.song_data.loc[track_id]
                    p = Process(target=self.generate_worker, args=(data, man_dict))
                    pool.append(p)
                    p.start()
                    first += split_val
                    second += split_val
                data = indexes[first:]
                p = Process(target=self.generate_worker, args=(data, man_dict))
                pool.append(p)
                p.start()
                for proc in pool:
                    proc.join()
                print("Beginning Write to File")
                for key in man_dict.keys():
                    self.radsData[key] = man_dict[key]
            dump(self.radsData, open('songs_{}_{}_'.format(1, len(indexes)//x) + pickle_filename, 'wb'))
                

    def generate_worker(self, data, mgr):
        for track_id in data:
            features = self.song_data.loc[track_id]
            mgr[track_id] = {}
            users = []
            for user_id in self.userModels.keys():
                current_model = self.userModels[user_id]
                value = current_model.decision_function([features])
                if value < 0:
                    users.append((user_id, -1*value))
            mgr[track_id] = users
        
    

if __name__ == "__main__":
    song_feature_files = ['musicbrainz-data/song_features.csv']
    user_history_files = ['musicbrainz-data/userid-trackid-1.csv', 'musicbrainz-data/userid-trackid-2.csv']
    model = RADS(song_feature_files, user_history_files, 'user_models.p', 'user_histories.p')
    model.generate('songs_1_7_rads_data.p')
    model.get_output(20, all=False)