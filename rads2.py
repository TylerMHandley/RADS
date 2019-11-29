from anomalyDetection import AnomalyDetection
from os.path import exists
from pickle import load, dump
from pandas import read_csv
from multiprocessing import Process, Manager, Pool
from auralist import Auralist

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
            print('Loading anomaly list from {}'.format(pickle_filename))
            self.radsData = load(open(pickle_filename, 'rb'))
        else:
            x = 50
            indexes = list(self.userModels.keys())
            y = len(indexes)//x
            indexes = indexes[:y]
            total_ids = len(indexes)
            split_val = total_ids//8
            # self.radsData = dict.fromkeys(indexes)      
            # with Manager() as manager:
                # man_dict = manager.dict(self.radsData)
                # pool = []
            data = []
            count = 0
            first = 0
            second = split_val
            while(second<total_ids):
                data.append(indexes[first:second])
                first += split_val
                second += split_val
                # p = Process(target=self.generate_worker, args=(data, man_dict))
                # pool.append(p)
                # p.start()
            data.append(indexes[first:])
            # p = Process(target=self.generate_worker, args=(data, man_dict))
            # pool.append(p)
            # p.start()
            # for proc in pool:
                # proc.join()
            p = Pool(processes=2)
            results = p.map(self.generate_worker, data)
            print("Beginning Write to File")
            self.radsData = {}
            for i in results:
                self.radsData.update(dict(i))
            # for key in man_dict.keys():
                # self.radsData[key] = man_dict[key]
            dump(self.radsData, open(pickle_filename, 'wb'))

    def get_output(self,all=True):
            results = []
            users = []
            if all:
                for user in self.radsData.keys():
                    partial = sorted(self.radsData[user], key=lambda x: x[1])
                    #results[user] = list(map(lambda x: x[0], partial))
                    results.append(list(map(lambda x: x[0], partial)))
                    users.append(user)
            else:
                partial = sorted(self.radsData['user_000001'], key=lambda x: x[1])
                #results['user_000001'] = list(map(lambda x: x[0], partial))
                results.append(list(map(lambda x: x[0], partial)))
                users.append['user_000001']
            return results, users      

    def generate_worker(self, data):
        results = []
        for user_id in data:
            current_model = self.userModels[user_id]
            tracks = []
            for track_id in self.song_data.index:
                features = self.song_data.loc[track_id]
                value = current_model.decision_function([features])
                tracks.append((track_id, -1*1/value))
            results.append((user_id, tracks))
        return results
        
    

if __name__ == "__main__":
    song_feature_files = ['musicbrainz-data/song_features.csv']
    user_history_files = ['musicbrainz-data/train1.csv', 'musicbrainz-data/train2.csv']
    model = RADS(song_feature_files, user_history_files, 'user_models.p', 'user_histories.p')
    model.generate('rads_data.p')
    anomaly_results = model.get_output(all=False)
    print(len(anomaly_results))
    aur = Auralist()
    for i in range(len(anomaly_results)):
        found_songs = [] 
        for j in range(len(anomaly_results[i])):
            x = aur.trackid2index.get(anomaly_results[i][j], -1)
            if x != -1:
                found_songs.append(x)
        anomaly_results[i] = found_songs
        # anomaly_results[i] = list(map(lambda x: aur.trackid2index[x], anomaly_results[i]))
    basic_aur_results = aur.basic_auralist('user_000001')
    print(len(basic_aur_results), len(anomaly_results[0]))
    final = aur.linear_interpolation((0.7, basic_aur_results),(0.3, anomaly_results[0]))
    with open('single_rads_result.p', 'wb+') as output:
        dump(final, output)
