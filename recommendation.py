#!/usr/bin/env python
from numpy import sum as npsum
from numpy import matmul, norm, reciprocal
from math import log
# from auralist import Auralist
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



def getMetrics(user_histories, top_n_songs, song_populary, n=20):
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


        

if __name__ == '__main__':
    # aur = Auralist()

