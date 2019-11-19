from pandas import read_csv
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
import pickle
from sys import argv
from numpy import matmul, divide, reciprocal, zeros, log2, array, sum as npsum
from numpy.linalg import norm
from sklearn.linear_model import LinearRegression
from os.path import isfile

class Auralist:
    def __init__(self, num_topics=100):
        print('Loading listening history, pt. 1...', end='\r')
        hist_pt1 = read_csv('musicbrainz-data/userid-trackid-1.csv')
        print('Loading listening history, pt. 2...')
        hist_pt2 = read_csv('musicbrainz-data/userid-trackid-2.csv')
        total_hist = hist_pt1.append(hist_pt2)
        grouped_hist = total_hist.groupby('track_id')
        self.total_hist = total_hist.set_index('user_id')

        if isfile('pickles/raw.pickle'):
            print('Loading raw corpus (from pickle)...')
            with open('pickles/raw.pickle', 'rb') as f:
                self.corpus_raw = pickle.load(f)
        else:
            print('Loading raw corpus...')
            self.corpus_raw = self.create_raw_pickle(grouped_hist)

        if isfile('pickles/bow.pickle'):
            print('Loading user BOW representations (from pickle)...')
            with open('pickles/bow.pickle', 'rb') as f:
                self.bow_users = pickle.load(f)
        else:
            print('Loading user BOW representations...')
            self.bow_users = self.create_bow_pickle()
        
        print('Creating track id -> index pairing...')
        track_ids = grouped_hist.apply(lambda x: x.name)
        indexes = list(range(0, len(track_ids)))
        # Consider replacing this with bidict (downloaded library)
        self.trackid2index = dict(zip(track_ids, indexes))
        self.index2trackid = dict(zip(indexes, track_ids))

        print('Counting # of unique listeners for every song...')
        self.popularity = array([len(set(user_list)) for user_list in self.corpus_raw]).reshape(-1, 1)

        if isfile('pickles/topic.pickle'):
            print('Loading LDA matrix (from pickle)...')
            with open('pickles/topic.pickle', 'rb') as f:
                self.topic_composition = pickle.load(f)
        else:
            print('Training LDA...')
            self.topic_composition = self.train_lda(num_topics)

        print('Calculating listener diversity array...')
        self.listener_div = self.listener_diversity_rankings(self.topic_composition, self.popularity)

    def lda_similarity(self, topic_comp, user_tc):
        # The code below does the following, but slightly faster:
        # tc_norm = norm(topic_comp, axis=1, keepdims=True)
        # user_norm = norm(user_tc, axis=1, keepdims=True)
        # topic_com *= reciprocal(tc_norm)
        # user_tc *= reciprocal(user_norm)
        # mult = matmul(topic_comp, user_tc.T)
        # return np.sum(mult, axis=1, keepdims=True)
        return npsum(matmul(topic_comp * reciprocal(norm(topic_comp, axis=1, keepdims=True)), \
            (user_tc * reciprocal(norm(user_tc, axis=1, keepdims=True))).T), axis=1)

    def basic_auralist(self, user_id):
        # Get topic composition for user history matrix
        user_topic_composition = self.get_user_topic_comp(user_id)

        rec_vals = zeros(len(self.topic_composition))
        # Sum over chunks at a time, because otherwise it takes up way too much memory
        for i in range(0, len(user_topic_composition), 500):
            rec_vals += self.lda_similarity(self.topic_composition, user_topic_composition[i: i+500])
        # Each element of the array corresponds to the ranking score for
        # a particular track
        # This gives you the indices that would sort the array
        # this is what we want, since the indices correspond to track IDs
        return rec_vals.argsort()[::-1]

    def get_user_topic_comp(self, user_id):
        """Gets all track IDs in a user's listening history
        param:
            user_id: user in dataset to get track IDs for
        """
        user_hist = self.total_hist.loc[user_id]
        track_ids = sorted([self.trackid2index[key] for key in user_hist['track_id']])
        return self.topic_composition[track_ids]

    def listener_diversity_rankings(self, topic_composition, popularity):
        # Calculate equation for Listener Diversity
        list_div_basic = -1 * npsum((topic_composition * log2(topic_composition)), axis=1)
        # Calculate linear regression for Offset_pop(i)
        lr = self.train_listener_diversity_lr(popularity, list_div_basic)
        # Calculate Listener Diversity'
        list_div = list_div_basic - lr.predict(popularity)
        # Each element of the array corresponds to the ranking score for
        # a particular track
        # This gives you the indices that would sort the array
        # this is what we want, since the indices correspond to track IDs
        return list_div.argsort()[::-1]

    def train_listener_diversity_lr(self, popularity, diversity):
        # Calculate linear regression for Offset_pop(i)
        return LinearRegression().fit(popularity, diversity)   

    def train_lda(self, num_topics):
        lda = LdaModel(self.bow_users, num_topics=num_topics)
        # Holds the topic composition vector for each track id
        # I'm not sure whether we should use this (which gives weights)
        # or whether we should use get_document_topics (slower) that
        # gives probabilities. In theory they should be the same
        topic_composition = lda.inference(self.bow_users)[0] # returns a tuple, 2nd value is usually None
        with open('topic.pickle', 'wb') as f:
            pickle.dump(topic_composition, f)
        return topic_composition

    def create_bow_pickle(self):
        dictionary = Dictionary(self.corpus_raw)
        # In our context, the documents are songs, and the users are words
        # The bag of words representation essentially counts the number of times
        # a user appeared in a certain song's listening history
        bow_users = []
        for users_l in self.corpus_raw:
            bow_users.append(dictionary.doc2bow(users_l))
        # Save to a pickle for easier loading in the future
        with open('pickles/bow.pickle', 'wb') as f:
            pickle.dump(bow_users, f)
        return bow_users

    def create_raw_pickle(self, grouped_hist):
        # corpus_raw will be a list of lists. The lists will be lists of users
        # who listened to a particular song. We need to do this conversion because
        # Gensim expects a list as input, afaik
        corpus_raw = []
        for track_id, _ in grouped_hist:
            corpus_raw.append(grouped_hist.get_group(track_id)['user_id'].to_list())
        # Save to a pickle for easier loading in the future
        with open('pickles/raw.pickle', 'wb') as f:
            pickle.dump(corpus_raw, f)
        return corpus_raw

if __name__ == "__main__":
    aur = Auralist()
    print(aur.basic_auralist('user_000001'))