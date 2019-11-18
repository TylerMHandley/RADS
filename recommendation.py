#!/usr/bin/env python
from sklearn.metrics.pairwise import cosine_similarity
from numpy import sum as npSum
from math import log
'''
    This file will perform the reccomendations on the output of our algorithm
'''

def unserendipity(user_histories, top_n_songs, n):
    total_num_users = len(user_histories)
    unserendipity = 0
    for user in user_histories.keys():
        history = user_histories[user]
        recommendation = top_n_songs[user]
        sim = cosine_similarity(history, recommendation)
        unserendipity += 1/(total_num_users*len(history))*(npSum(sim)/n)
    return unserendipity

def novelty(user_histories, top_n_songs, n, song_populary):
    total_num_users = len(user_histories)
    novelty = 0
    for user in user_histories.keys():
        history = user_histories[user]
        recommendation = top_n_songs[user]
        novelty += sum(map(lambda x: log(song_populary[x]), recommendation))
        # for song in recommendation:
        #     novelty += log(song_populary[song])/n
    return novelty/total_num_users

def getMetrics(user_histories, top_n_songs, song_populary, n=20):
    total_num_users = len(user_histories)
    unserendipity = 0
    novelty = 0
    for user in user_histories.keys():
        history = user_histories[user]
        recommendation = top_n_songs[user]
        sim = cosine_similarity(history, recommendation)
        unserendipity += 1/(total_num_users*len(history))*(npSum(sim)/n)
        novelty += sum(map(lambda x: log(song_populary[x]), recommendation))
        # for song in recommendation:
        #     novelty += log(song_populary[song])/n
    novelty = novelty/total_num_users
    return novelty, unserendipity


        

