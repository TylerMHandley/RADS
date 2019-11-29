#!/usr/bin/env python
from numpy import sum as npsum
from numpy import matmul, reciprocal
from numpy.linalg import norm
from math import log
from pandas import read_csv 
from auralist import Auralist
from rads2 import RADS
'''
    This file will perform the reccomendations on the output of our algorithm
'''

# def unserendipity(user_histories, top_n_songs, n):
#     total_num_users = len(user_histories)
#     unserendipity = 0
#     for user in user_histories.keys():
#         history = user_histories[user]
#         recommendation = top_n_songs[user]
#         sim = cosine_similarity(history, recommendation)
#         unserendipity += 1/(total_num_users*len(history))*(npSum(sim)/n)
#     return unserendipity

# def novelty(user_histories, top_n_songs, n, song_populary):
#     total_num_users = len(user_histories)
#     novelty = 0
#     for user in user_histories.keys():
#         history = user_histories[user]
#         recommendation = top_n_songs[user]
#         novelty += sum(map(lambda x: log(song_populary[x]), recommendation))
#         # for song in recommendation:
#         #     novelty += log(song_populary[song])/n
#     return novelty/total_num_users
def similarity(self, topic_comp, user_tc):
        """Compute the LDA similarity between each row of two different matrices
        Follows the definition of LDA Similarity in the Auralist paper
        params:
            topic_comp: total topic composition matrix
                This will almost always be self.topic_composition
                but it is open to other uses
            user_tc: User's topic composition matrix
                Topic composition matrix that corresponds to the user's
                subset of listened-to songs. General way to get this is to
                call get_user_topic_comp()
        returns: an array, where each index corresponds to a song's similarity to
            the subset of songs in user_tc
        """
        return npsum(matmul(topic_comp * reciprocal(norm(topic_comp, axis=1, keepdims=True)), \
                            (user_tc * reciprocal(norm(user_tc, axis=1, keepdims=True))).T), axis=1)



def getMetricsAll(user_history, top_n_songs, song_populary, n=20):
    total_num_users = len(user_histories)
    unserendipity = 0
    novelty = 0
    for user in user_histories['user_id'].unique():
        history = user_histories.loc[user]
        recommendation = top_n_songs[user]
        sim = similarity(history, recommendation)
        unserendipity += 1/(total_num_users*len(history))*(npSum(sim)/n)
        novelty += sum(map(lambda x: log(song_populary[x]), recommendation))
        # for song in recommendation:
        #     novelty += log(song_populary[song])/n
    novelty = novelty/total_num_users
    return novelty, unserendipity

def getMetrics(history, recommendation, song_populary, total_num_users, n=20):
    sim = similarity(history, recommendation)
    unserendipity = 1/(total_num_users*n)*(npSum(sim)/n)
    novelty = sum(map(lambda x: log(song_populary[x]), recommendation))
    return novelty, unserendipity


        

if __name__ == '__main__':
    aur = Auralist()
    listener_diversity = aur.listener_diversity_rankings
    song_populary = aur.popularity_count/sum(aur.popularity_count)
    test_data = read_csv('musicbrainz-data/test.csv', index_col='user_id')
    song_feature_files = ['musicbrainz-data/song_features.csv']
    user_history_files = ['musicbrainz-data/train1.csv', 'musicbrainz-data/train2.csv']
    model = RADS(song_feature_files, user_history_files, 'user_models.p', 'user_histories.p')
    model.generate('rads_data.p')
    anomaly_results, users = model.get_output()
    for i in range(len(anomaly_results)):
        found_songs = []
        for j in range(len(anomaly_results[i])):
            x = aur.trackid2index.get(anomaly_results[i][j], -1)
            if x != -1:
                found_songs.append(x)
        anomaly_results[i] = found_songs
    total_users = len(users)
    n = 100
    rads_novelty = 0
    rads_serendipity = 0
    aur_novelty = 0
    aur_serendipity = 0
    for user in users:
        basic_aur_results = aur.basic_auralist(user)
        rads =  aur.linear_interpolation((0.7, basic_aur_results),(0.3, anomaly_results[0]))
        declustering_ranking = aur.declustering_ranking(user)
        auralist = aur.linear_interpolation((0.7, basic_aur_results),(0.15, listener_diversity),(0.15, declustering_ranking))
        x, y = getMetrics(test_data[user], rads[-1*n:], song_populary, total_users, n)
        rads_novelty += x
        rads_serendipity += y
        x, y = getMetrics(test_data[user], auralist[-1*n:], song_populary, total_users, n)
        aur_novelty += x
        aur_serendipity += y
    rads_novelty/= total_users
    aur_novelty /= total_users
    print("Rads Novelty and Serendipity: {} {}".format(rads_novelty, rads_serendipity))
    print("Auralist Novelty and Serendipity: {} {}".format(aur_novelty, aur_serendipity))