from anomalyDetection import AnomalyDetection
from os.path import exists
from pickle import load, dump
from pandas import read_csv
import tqdm
from multiprocessing import Process, Manager


def loadData(filenames, index_column):
        data = read_csv(filenames[0], index_col=index_column)
        if len(filenames) > 1:
            for i in range(1, len(filenames)):
                extra_part = read_csv(filenames[i], index_col=index_column)
                data.append(extra_part)
        return data

class RADS:
    def __init__(self, song_files, user_files, pickle_filename=None):
        self.radsData = None
        self.userModels = None
        self.song_data = loadData(song_files, 'track_id')
        self.user_history_files = user_files
        self.pickle_filename = pickle_filename
        self.getSVMs()

    def getSVMs(self):
        if exists(self.pickle_filename):
            self.userModels = load(open(self.pickle_filename, 'rb'))
            print("Pickled State Loaded from {}".format(self.pickle_filename))
        else:
            print("Pickled file not found, generating SVMs...")
            user_data = loadData(self.user_history_files, 'user_id')
            anomalyDetector = AnomalyDetection(self.song_data, user_history_data)
            anomalyDetector.build(self.pickle_filename)
            self.userModels = anomalyDetector.models

    def generate(self, pickle_filename):
        if exists(pickle_filename):
            self.radsData = load(open(pickle_filename, 'rb'))
        else:
            indexes = list(self.song_data.index)
            total_ids = len(indexes)
            split_val = total_ids//12
            self.radsData = dict.fromkeys(indexes)      
            with Manager() as manager:
                man_dict = manager.dict()
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
                self.rads_data = man_dict
        dump(self.radsData, open(pickle_filename, 'wb'))

                

    def generate_worker(self, data, mgr):
        for track_id in data:
            features = self.song_data.loc[track_id]
            mgr[track_id] = {}
            for user_id in self.userModels.keys():
                current_model = self.userModels[user_id]
                value = current_model.decision_function([features])
                if value[0] < 0:
                    mgr[track_id][user_id] = -1 * value[0]
            
        
    

if __name__ == "__main__":
    song_feature_files = ['musicbrainz-data/song_features.csv']
    user_history_files = ['musicbrainz-data/userid-trackid-1.csv', 'musicbrainz-data/userid-trackid-2.csv']
    model = RADS(song_feature_files, user_history_files, 'user_models.p')
    model.generate('rads_data.p')