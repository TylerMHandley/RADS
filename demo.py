from auralist import Auralist
from pickle import load
from pandas import read_csv

if __name__ == "__main__":
    n = 3
    with open('pickles/user_recommendations/user_000001_rec.p', 'rb') as input_recs:
        user_recs = load(input_recs)
    rads_rec = user_recs[0][-n:]
    aur_rec = user_recs[1][-n:]
    aur = Auralist()
    history = aur.total_hist
    #history = read_csv('musicbrainz-data/train1.csv', index_col='user_id')
    history = history.loc['user_000001']
    counted_history = history['track_id'].value_counts()
    top_songs = counted_history.iloc[:10]
    print(top_songs.idxmax())
    for i in range(n):
        print("Rads recommendation {} is {}".format(n-i, aur.index2trackid[rads_rec[i]]))
        print("Aur recommendation {} is {}".format(n-i, aur.index2trackid[aur_rec[i]]))