# Data Mining Data

In order to actually fit the data on github, I had to split it into a couple files and compress it. To use the data, uncompress the zip files, then do the following:

```python
from pandas import read_csv

# song_features will have the features of each track id
song_features = read_csv('song_features.csv', index_col='track_id')
# you can query a track's features by doing:
# song_feats.loc[*track_id*]
# For example, song_feats.loc['0024d72c-136f-49f2-9078-ce4b39b94d3f']

# listen_hist will have the listening histories for every user
part1 = read_csv('userid-trackid-1.csv', index_col='user_id')
part2 = read_csv('userid-trackid-2.csv', index_col='user_id')
listen_hist = part1.append(part2)
# Query a user's history by doing:
# listen_hist.loc[*user_id*]
# For example, listen_hist.loc['user_000001']
```
