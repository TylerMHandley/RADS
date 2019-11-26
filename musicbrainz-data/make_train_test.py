from pandas import read_csv, DataFrame

pt1 = read_csv('userid-trackid-1.csv', index_col='user_id')
pt2 = read_csv('userid-trackid-2.csv', index_col='user_id')
total_hist = pt1.append(pt2)

train = DataFrame()
test = DataFrame()

for user in total_hist.index.unique():
    user_total = total_hist.loc[user].reset_index()
    user_train = user_total.sample(frac=0.8, random_state=69)
    user_test = user_total.loc[~user_total.index.isin(user_train.index)]
    # There aren't enough records to create a testing set for this user, skip them
    if len(user_test) == 0:
        print('Done with {}'.format(user), end='\r')
        continue
    train = train.append(user_train.sort_index())
    test = test.append(user_test.sort_index())
    print('Done with {}'.format(user), end='\r')

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)