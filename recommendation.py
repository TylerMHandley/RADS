#!/usr/bin/env python
from numpy import sum as npsum
from numpy import matmul, reciprocal, array, zeros
from numpy.linalg import norm
from math import log
from pandas import read_csv, DataFrame 
from auralist import Auralist
from rads2 import RADS
from os.path import exists
from pickle import load, dump
from tqdm import tqdm
from time import time
'''
    This file will perform the reccomendations on the output of our algorithm
'''

def cosim(i,j,aur):
    i_users = set(aur.corpus_raw[i])
    j_users = set(aur.corpus_raw[j])
    ret_val = len(i_users.intersection(j_users))
    ret_val /= (len(i_users))**(1/2) * (len(j_users))**(1/2)
    return ret_val

def getRecall(withheld, rec_rads, rec_aur):
    rads_count = 0
    aur_count = 0
    for song_index in withheld:
        if song_index in rec_rads:
            rads_count += 1
        if song_index in rec_aur:
            aur_count +=1
    return rads_count/len(withheld), aur_count/len(withheld)

def getPopularity(corpus):
    popularity = zeros((len(corpus), 1))
    pop_sum = 0
    for i in range(len(corpus)):
        x = len(corpus[i])
        pop_sum += x
        popularity[i] = x
    popularity /= pop_sum
    return popularity
        

if __name__ == '__main__':
    aur = Auralist()
    song_populary = getPopularity(aur.corpus_raw)
    test_data = read_csv('musicbrainz-data/test.csv', index_col='user_id')
    song_feature_files = ['musicbrainz-data/song_features.csv']
    user_history_files = ['musicbrainz-data/train1.csv', 'musicbrainz-data/train2.csv']
    model = RADS(song_feature_files, user_history_files, 'user_models.p', 'user_histories.p')
    total_users = test_data.index.nunique()
    rads_recall = 0
    users =  test_data.index.unique()
    aur_recall = 0
    
    n = 100
    rad_rec = []
    aur_rec = []
    temp_users = []

    lam1 = 0.15
    lam2 = 0.15
    # lam3 = 0.15
    # count = 0
    # rads_time = 0
    # auralist_time = 0
    # basic_time = 0
    print("Lam1: {} Lam2: {}".format(lam1, lam2))
    for user in tqdm(users[:-1], ascii=True, leave=False):
        filename = 'pickles/user_rankings/{}_rec.p'.format(user)

        if exists(filename):
            # print("Loading {} from pickle".format(user))
            temp_users.append(user)
            with open(filename, 'rb') as input_file:
                temp = load(input_file)
                basic_aur_results = temp[0]
                anomaly_indexes = temp[1]
                listener_diversity = temp[2]
                declustering_ranking = temp[3]
                indices = aur.get_indices_from_basic_auralist(basic_aur_results)
                rads = aur.linear_interpolation(indices, (1 - (lam1 + lam2), basic_aur_results),((lam1 + lam2), anomaly_indexes))
                auralist = aur.linear_interpolation(indices, (1 - (lam1 + lam2), basic_aur_results),(lam1, listener_diversity), (lam2, declustering_ranking))
                # auralist = aur.linear_interpolation(indices, (1 - (lam1 + lam2 + lam3), basic_aur_results),(lam1, listener_diversity), (lam2, declustering_ranking), (lam3, anomaly_indexes))
                rad_rec.append(rads[-n:])
                aur_rec.append(auralist[-n:])
                user_history = []
                for _, i in test_data.loc[user].iterrows():
                    user_history.append(aur.trackid2index.get(i['track_id'], -1))  
                x, y = getRecall(user_history, rads[-n:], auralist[-n:])
                rads_recall += x
                aur_recall += y
        else:
            user_history = []
            user_data = test_data.loc[user]
            if not isinstance(user_data, DataFrame):
                # print('User {} did not have a large enough test set ({})'.format(user, len(user_data)))
                continue
            temp_users.append(user)
            for _, i in user_data.iterrows():
                user_history.append(aur.trackid2index.get(i['track_id'], -1))   
            # basic_start = time()
            basic_aur_results = aur.basic_auralist(user)
            # basic_stop = time()
            # rads_time += basic_stop - basic_start
            # basic_time += basic_stop - basic_start
            # auralist_time += basic_stop - basic_start
            indices = aur.get_indices_from_basic_auralist(basic_aur_results)
            candidate_id = []
            for i in indices:
                candidate_id.append(aur.index2trackid[i])
            # start = time()
            anomaly_results = model.generate_worker(user, candidate_id)
            # stop = time()
            # rads_time += stop - start
            anomaly_indexes = []
            for i in anomaly_results:
                anomaly_indexes.append(aur.trackid2index[i])
            rads =  aur.linear_interpolation(indices, (1 - (lam1 + lam2), basic_aur_results),((lam1 + lam2), anomaly_indexes))
            # start = time()
            declustering_ranking = aur.declustering(user, indices)
            listener_diversity = aur.listener_diversity_from_indexes(indices)
            # stop = time()
            # auralist_time += stop - start
            auralist = aur.linear_interpolation(indices, (1 - (lam1 + lam2), basic_aur_results),(lam1, listener_diversity), (lam2, declustering_ranking))
            x, y = getRecall(user_history, rads[-1*n:], auralist[-1*n:])
            rads_recall += x
            aur_recall += y
            rad_rec.append(rads[-1*n:])
            aur_rec.append(auralist[-1*n:])
            with open(filename, 'wb+') as output:
                dump([basic_aur_results, anomaly_indexes, listener_diversity, declustering_ranking], output)
    # print("Basic Collaborative took {} seconds per user".format(auralist_time/count))
    # print("Rads took {} seconds per user".format(rads_time/count))
    # print("Auralist took {} seconds per user".format(auralist_time/count))
    
    
    print('The {}-Recall for Rads is {}'.format(n, rads_recall/total_users))
    print('The {}-Recall for Auralist is {}'.format(n, aur_recall/total_users))
    aur_novelty = 0
    aur_serendipity = 0
    rads_novelty = 0
    rads_serendipity = 0

    total_users = len(temp_users)

    for i in range(total_users):
        for j in range(n):
            r = rad_rec[i][j]
            a = aur_rec[i][j]
            aur_novelty += log(song_populary[a])/n
            rads_novelty += log(song_populary[r])/n
    print('The {}-Novelty for Rads is {}'.format(n, -1*rads_novelty/total_users))
    print('The {}-Novelty for Auralist is {}'.format(n, -1*aur_novelty/total_users))
    
    for dex, user in enumerate(temp_users):
        history = aur.total_hist.loc[user]
        history = [aur.trackid2index[key] for key in history['track_id'].unique()]
        aur_temp = 0
        rads_temp = 0
        for i in history:
            for j in aur_rec[dex]:
                aur_temp += cosim(i,j,aur)/n
            for j in rad_rec[dex]:
                rads_temp += cosim(i,j,aur)/n
        rads_serendipity += 1/(total_users * len(history)) * rads_temp
        aur_serendipity +=  1/(total_users * len(history)) * aur_temp
    
    print('The {}-Unserendipity for Rads is {}'.format(n, rads_serendipity))
    print('The {}-Unserendipity for Auralist is {}'.format(n, aur_serendipity))

        
        
        
