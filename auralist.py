from pandas import read_csv
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
import pickle
from numpy import delete, nonzero, matmul, reciprocal, zeros, log2, array, multiply, arange, copy, sum as npsum
from numpy.linalg import norm
from sklearn.linear_model import LinearRegression
from os.path import isfile


class Auralist:
    def __init__(self, num_topics=100):
        """Constructor. Sets up all needed variables for prediction
        Some variables will be loaded from pickle files, if they're available.
        """
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
        self.popularity_count = array([len(set(user_list)) for user_list in self.corpus_raw]).reshape(-1, 1)

        if isfile('pickles/topic.pickle'):
            print('Loading LDA matrix (from pickle)...')
            with open('pickles/topic.pickle', 'rb') as f:
                self.topic_composition = pickle.load(f)
        else:
            print('Training LDA...')
            self.topic_composition = self.train_lda(num_topics)

        # print('Calculating listener diversity array...')
        # self.listener_diversity_rankings = self.listener_diversity(self.topic_composition, self.popularity_count)
        print("Start Declustering")

    def community_aware_ranking(self, user_id, diversity_lambda):
        """Return ranks for Community-Aware Auralist, from page 6 of the paper
        params:
            user_id: ID of the user to predict recommendations for. Used in basic Auralist
            diversity_lambda: Linear interpolation constant for the listener diversity ranking
        returns:
            A ranking that combines the output of the basic Auralist and listener diversity
            submodules, given as a numpy array of indices
        """
        basic = (1 - diversity_lambda, self.basic_auralist(user_id))
        diversity = (diversity_lambda, self.listener_diversity_rankings)
        return self.linear_interpolation(basic, diversity)

    def lda_similarity(self, topic_comp, user_tc):
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
        """Perform basic auralist, as described in the Auralist paper
        This is module 1 of Auralist
        params:
            user_id: ID of the user to predict recommendations for
        returns: indexes of the track IDs to recommend, where 0 is the least recommended,
            element at end is most recommended
        """
        # Get topic composition for user history matrix
        user_topic_composition = self.get_user_topic_comp(user_id)
        rec_vals = zeros(len(self.topic_composition))
        # Sum over chunks at a time, because otherwise it takes up way too much memory
        for i in range(0, len(user_topic_composition), 500):
            rec_vals += self.lda_similarity(self.topic_composition, user_topic_composition[i: i + 500])
        # Each element of the array corresponds to the ranking score for
        # a particular track
        # This gives you the indices that would sort the array
        # this is what we want, since the indices correspond to track IDs
        return rec_vals.argsort()

    def get_user_topic_comp(self, user_id):
        """Gets all track IDs in a user's listening history, returns topic composition
        param:
            user_id: user in dataset to get track IDs for
        """
        user_hist = self.total_hist.loc[user_id]
        track_ids = sorted([self.trackid2index[key] for key in user_hist['track_id'].unique()])
        return self.topic_composition[track_ids]

    def get_user_track_ids(self, user_id):
        """Gets all track IDs in a user's listening history
        param:
            user_id: user in dataset to get track IDs for
        """
        user_hist = self.total_hist.loc[user_id]
        track_ids = sorted([self.trackid2index[key] for key in user_hist['track_id'].unique()])
        return track_ids

    def listener_diversity(self, topic_composition, popularity):
        """Calculate rankings for module 2 of Auralist.
        params:
        topic_comp: total topic composition matrix
                This will almost always be self.topic_composition
                but it is open to other uses
        popularity: array or list of popularity measures for each song. In the current
            implementation, popularity is count of how many unique users listened to a
            song
        returns: List of indexes sorted by listener diversity
        """
        # Calculate equation for Listener Diversity
        # Take log of topic_composition
        tc_copy = copy(topic_composition)
        mask = tc_copy != 0
        tc_copy[mask] = log2(tc_copy[mask])
        # Multiply element-wise, then sum over rows and negate
        list_div_basic = -1 * npsum((topic_composition * tc_copy), axis=1)
        # Calculate linear regression for Offset_pop(i)
        lr = self.train_listener_diversity_lr(popularity, list_div_basic)
        # Calculate Listener Diversity'
        list_div = list_div_basic - lr.predict(popularity)
        # Each element of the array corresponds to the ranking score for
        # a particular track
        # This gives you the indices that would sort the array
        # this is what we want, since the indices correspond to track IDs
        return list_div.argsort()

    def user_pairs_for_similarity(self, topic_comp, user_tc):
        return matmul(topic_comp * reciprocal(norm(topic_comp, axis=1, keepdims=True)), \
                      (user_tc * reciprocal(norm(user_tc, axis=1, keepdims=True))).T)

    def declustering(self, user_id):
        '''Performs the declustering module (4.2.2)
        :param user_id: to get topic composition info for a particular user, and then
        apply declustering for that user

        :return: returns: indexes of the track IDs to recommend (via clustering score),
        where 0 is the least recommended, element at end is most recommended
        '''
        # Get topic composition for user history matrix
        user_topic_composition = self.get_user_topic_comp(user_id)
        user_track_ids = self.get_user_track_ids(user_id)
        # matrix where entry i,j contains the sim(i,j) result for songs i,j in the user's comp.
        abc = self.user_pairs_for_similarity(user_topic_composition, user_topic_composition)
        de = self.user_pairs_for_similarity(self.topic_composition, user_topic_composition)
        print(de.shape)
        new_len = len(de) # treat as full length now... - len(user_track_ids)
        #518924, 289541, 466981
        size = len(abc[0])
        total = 0
        count = 0
        # Calculate avgSim, iterating through lower triangle of symmetric matrix abc
        for i in range(size):
            for j in range(i + 1):
                total += abc[i, j]
                count += 1
        avgSim = total / count
        # make a copy of abc, fill entries with 1 if their num > avgSim, 0 otherwise
        abc_bool = copy(abc)
        for i in range(size):
            for j in range(i + 1):
                if abc[i, j] > avgSim:
                    abc_bool[i, j] = 1
                    abc_bool[j, i] = 1
                else:
                    abc_bool[i, j] = 0
                    abc_bool[j, i] = 0
        # cluster_list to obtain result of declustering
        cluster_list = zeros((new_len, 2))
        # for item i in candidate set for user
        count=0
        for i in range(len(de)):
            #should print 0-520000 by intervals of 10000
            if i % 10000 == 0:
                print(i)
            #Get tracks user hasn't seen, while also using arr index for songid
            if i not in user_track_ids:
                edgeTotal = 0
                edgeCount = 0
                # neighbors of row i are indices with nonzero sim
                neighbors = nonzero(de[i])[0]
                #set index, taking into account 'deleted' indices
                #cluster_list[count][0] = i
                cluster_list[i][0] = i
                if len(neighbors) == 0:
                    #original: cluster_list[count][1] = 0
                    cluster_list[i][1] = 0
                else:
                    # Iterate through through upper triangle of matrix
                    # to account for each item pair j,k in neighbors
                    # But do not include item pairs (j,j)!
                    for j in range(len(neighbors) - 1):
                        temp_arr = abc_bool[neighbors[j]][neighbors[j + 1:]]
                        # "nxn matrix": Add to edgeTotal for nonzero (j,j+1),...,(j,n) in abc_bool
                        edgeTotal += npsum(temp_arr)
                        # Add to edgeCount by counting "j+1,j+2,...,n"
                        edgeCount += len(temp_arr)
                    # Get edgeTotal and edgeCount from ALL pairs
                    # Add song's trackid, followed by "clustering = edgeTotal/EdgeCount"
                    ###cluster_list[count][0] = de[i]

                    #original: cluster_list[count][1] = edgeTotal / edgeCount
                    cluster_list[i][1] = edgeTotal / edgeCount
                    count += 1
            else:
                #temp code for songs user listened to, towards len 520998 array
                cluster_list[i][0] = i
                cluster_list[i][1] = 0
        # Sort by clustering values
        cluster_list = cluster_list[cluster_list[:, 1].argsort()]
        return cluster_list[:, 0].astype(int)

    def bubble_aware_ranking(self, user_id, bubble_lambda):
        """Return ranks for Bubble-Aware Auralist, from page 6 of the paper
        params:
            user_id: ID of the user to predict recommendations for. Used in basic Auralist
            bubble_lambda: Linear interpolation constant for the listener declustering ranking
        returns:
            A ranking that combines the output of the basic Auralist and declustering
            submodules, given as a numpy array of indices
        """
        a = self.basic_auralist(user_id)
        print(a.shape)
        b = self.declustering(user_id)
        print(b.shape)
        basic = (1 - bubble_lambda, a)
        bubble = (bubble_lambda, b)
        return self.linear_interpolation(basic, bubble)

    def linear_interpolation(self, *rankings):
        """Perform linear interpolation of given rankings to give final recommendation result
        params:
            *rankings: variable number of (lambda, ranking list) pairings, ideally given as
                tuples. Lambda is the interpolation coefficient, and the ranking list is a list
                of indices in recommendation order for one of the sub-modules.
                As an example, if we have:
                basic = array([1, 0, 2, 3])
                diverse = array([2, 3, 1, 0])
                If we want basic to have 70% of the influence, and diverse to have 30% of the
                influence, we would call the function in the following way:
                linear_interpolation((0.7, basic), (0.3, diverse))
        returns:
            An interpolated ranking of the given index rankings. For example, if given the call
            in the example above, would return array([1, 2, 0, 3])
        """
        num_items = len(rankings[0][1])
        # This will hold the interpolated rankings. Each number can be thought of as the new
        # rank of the corresponding song. The indices->items mapping follows the same convention
        # as all other functions
        results = zeros(num_items)
        indices = arange(1, num_items + 1)
        for coeff, ranking_list in rankings:
            # Each ranking list is just a list of indices corresponding to certain tracks, so we
            # can index into results using this list
            results[ranking_list] += coeff * indices
        return results.argsort()

    def train_listener_diversity_lr(self, popularity, diversity):
        """Calculate linear regression for Offset_pop(i) from Auralist Paper
        params:
            popularity: array or list of popularity measures for each song. In the current
            implementation, popularity is count of how many unique users listened to a
            song
            diversity: Listener diversity matrix described in paper
        returns: Linear regression model fit w/ popularity as X and diversity as Y
        """
        return LinearRegression().fit(popularity, diversity)

    def train_lda(self, num_topics):
        """Train the Latent Dirichlet Allocation Model needed for Auralist
        The model will be saved to a pickle file if it isn't present
        params:
            num_topics: number of topics to generalize to for LDA model
        returns: topic composition matrix. This is a # songs x # topics matrix
            which contains values that correspond to how well a song is represented
            by the given topic
        """
        lda = LdaModel(self.bow_users, num_topics=num_topics)
        # Holds the topic composition vector for each track id
        # I'm not sure whether we should use this (which gives weights)
        # or whether we should use get_document_topics (slower) that
        # gives probabilities. In theory they should be the same
        topic_composition = lda.inference(self.bow_users)[0]  # returns a tuple, 2nd value is usually None
        # normalize topic composition matrix
        topic_composition /= topic_composition.sum(axis=1).reshape(-1, 1)
        # Remove small probabilities that are essentially noise (set them to 0)
        topic_composition = multiply(topic_composition, (topic_composition > 0.01).astype(int))
        with open('pickles/topic.pickle', 'wb') as f:
            pickle.dump(topic_composition, f)
        return topic_composition

    def create_raw_pickle(self, grouped_hist):
        """For each song, create list of users who listened to it. Save to pickle
        params:
            grouped_hist: track history dataset, grouped by track_id
        """
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

    def create_bow_pickle(self):
        """Create the bag of words representation of the dataset"""
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


if __name__ == "__main__":
    aur = Auralist()
    # print(aur.community_aware_ranking('user_000001', diversity_lambda=0.3))

    # print(aur.basic_auralist('user_000001'))

    # print(aur.get_user_track_ids('user_000001'))
    #print(aur.declustering('user_000001'))
    bub = aur.bubble_aware_ranking('user_000001',0.5)
    print(bub)
    # print(aur.bubble_aware_ranking('user_000001',1))

    # print("First:")
    # print(aur.basic_auralist('user_000001'))
    # print(aur.basic_auralist('user_000001').shape)
