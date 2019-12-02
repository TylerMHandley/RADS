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
'''
    This file will perform the reccomendations on the output of our algorithm
'''
# def cosim(x, y):
#     count = 0
#     for song_index in x:
#         if song_index in x:
#             count+=1
#     d1 = len(x)**(1/2)
#     d2 = len(y)**(1/2)
#     return count/(d1*d2)


# def getMetricsAll(user_history, top_n_songs, song_populary, n=20):
#     total_num_users = len(user_histories)
#     unserendipity = 0
#     novelty = 0
#     for user in user_histories['user_id'].unique():
#         history = user_histories.loc[user]
#         recommendation = top_n_songs[user]
#         sim = similarity(history, recommendation)
#         unserendipity += 1/(total_num_users*len(history))*(npSum(sim)/n)
#         novelty += sum(map(lambda x: log(song_populary[x]), recommendation))
#         # for song in recommendation:
#         #     novelty += log(song_populary[song])/n
#     novelty = novelty/total_num_users
#     return novelty, unserendipity

# def getMetrics(history, recommendation, song_populary, total_num_users, n=20):
#     sim = similarity(history, recommendation)
#     unserendipity = 1/(total_num_users*n)*(npSum(sim)/n)
#     novelty = sum(map(lambda x: log(song_populary[x]), recommendation))
#     return novelty, unserendipity

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
    # candidate_index = aur.TODO
    # candidate_id = []
    # for i in candidate_index:
    #     candidate_id.append(aur.index2trackid[i])
    model = RADS(song_feature_files, user_history_files, 'user_models.p', 'user_histories.p')
    # model.generate('rads_data_alls.p')
    # anomaly_results, users = model.get_output()
    # for i in range(len(anomaly_results)):
    #     found_songs = []
    #     for j in range(len(anomaly_results[i])):
    #         x = aur.trackid2index.get(anomaly_results[i][j], -1)
    #         if x != -1:
    #             found_songs.append(x)
    #     anomaly_results[i] = found_songs
    total_users = test_data.index.nunique()
    rads_recall = 0
    users =  test_data.index.unique()
    aur_recall = 0
    
    n = 100
    rad_rec = []
    aur_rec = []
    for user in users:
        filename = 'pickles/user_recommendations/{}_rec.p'.format(user)
        if exists(filename):
            print("Loading {} from pickle".format(user))
            with open(filename, 'rb') as input_file:
                temp = load(input_file)
                rad_rec.append(temp[0])
                aur_rec.append(temp[1])
                user_history = []
                for _, i in test_data.loc[user].iterrows():
                    user_history.append(aur.trackid2index.get(i['track_id'], -1))  
                x, y = getRecall(user_history, temp[0], temp[1])
                rads_recall += x
                aur_recall += y
        else:
            user_history = []
            user_data = test_data.loc[user]
            if not isinstance(user_data, DataFrame):
                print('User {} did not have a large enough test set ({})'.format(user, len(user_data)))
                continue
            for _, i in user_data.iterrows():
                user_history.append(aur.trackid2index.get(i['track_id'], -1))   
            basic_aur_results = aur.basic_auralist(user)
            indices = aur.get_indices_from_basic_auralist(basic_aur_results)
            candidate_id = []
            for i in indices:
                candidate_id.append(aur.index2trackid[i])
            anomaly_results = model.generate_worker(user, candidate_id)
            anomaly_indexes = []
            for i in anomaly_results:
                anomaly_indexes.append(aur.trackid2index[i])
            rads =  aur.linear_interpolation(indices, (0.7, basic_aur_results),(0.3, anomaly_indexes))
            declustering_ranking = aur.declustering(user, indices)
            listener_diversity = aur.listener_diversity_from_indexes(indices)
            auralist = aur.linear_interpolation(indices, (0.7, basic_aur_results),(0.30, listener_diversity))
            x, y = getRecall(user_history, rads[-1*n:], auralist[-1*n:])
            rads_recall += x
            aur_recall += y
            rad_rec.append(rads[-1*n:])
            aur_rec.append(auralist[-1*n:])
            with open(filename, 'wb+') as output:
                dump([rads[-1*n:], auralist[-1*n:]], output)
    print('The {}-Recall for Rads is {}'.format(n, rads_recall/total_users))
    print('The {}-Recall for Auralist is {}'.format(n, aur_recall/total_users))
    aur_novelty = 0
    aur_serendipity = 0
    rads_novelty = 0
    rads_serendipity = 0
    for i in range(total_users):
        for j in range(n):
            r = rad_rec[i][j]
            a = aur_rec[i][j]
            aur_novelty += log(song_populary[a])/n
            rads_novelty += log(song_populary[r])/n
    print('The {}-Novelty for Rads is {}'.format(n, -1*rads_novelty/total_users))
    print('The {}-Novelty for Auralist is {}'.format(n, -1*aur_novelty/total_users))
    
    for dex, user in enumerate(users):
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

        
        
        
