#!/usr/bin/env python
from numpy import sum as npsum
from numpy import matmul, reciprocal, array
from numpy.linalg import norm
from math import log
from pandas import read_csv 
from auralist import Auralist
from rads2 import RADS
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


        

if __name__ == '__main__':
    aur = Auralist()
    listener_diversity = aur.listener_diversity_rankings
    song_populary = aur.popularity_count/sum(aur.popularity_count)
    test_data = read_csv('musicbrainz-data/test.csv', index_col='user_id')
    song_feature_files = ['musicbrainz-data/song_features.csv']
    user_history_files = ['musicbrainz-data/train1.csv', 'musicbrainz-data/train2.csv']
    model = RADS(song_feature_files, user_history_files, 'user_models.p', 'user_histories.p')
    model.generate('rads_data_alls.p')
    anomaly_results, users = model.get_output()
    for i in range(len(anomaly_results)):
        found_songs = []
        for j in range(len(anomaly_results[i])):
            x = aur.trackid2index.get(anomaly_results[i][j], -1)
            if x != -1:
                found_songs.append(x)
        anomaly_results[i] = found_songs
    total_users = len(users)
    rads_recall = 0
    
    aur_recall = 0
    
    n = 20
    rad_rec = []
    aur_rec = []
    for dex, user in enumerate(users):
        user_history = []
        for _, i in test_data.loc[user].iterrows():
            user_history.append(aur.trackid2index.get(i['track_id'], -1))   
        basic_aur_results = aur.basic_auralist(user)
        rads =  aur.linear_interpolation((0.7, basic_aur_results),(0.3, anomaly_results[dex]))
        declustering_ranking = aur.declustering(user)
        auralist = aur.linear_interpolation((0.7, basic_aur_results),(0.15, listener_diversity),(0.15, declustering_ranking))
        x, y = getRecall(user_history, rads[-1*n:], auralist[-1*n:])
        rads_recall += x
        aur_recall += y
        rad_rec.append(rads[-1*n:])
        aur_rec.append(auralist[-1*n:])
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
    print('The {}-Novelty for Rads is {}'.format(n, rads_novelty/total_users))
    print('The {}-Novelty for Auralist is {}'.format(n, aur_novelty/total_users))
    
    for dex, user in enumerate(users):
        history = aur.total_hist.loc[user_id]
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

        
        
        
