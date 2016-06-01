import csv
import datetime

import numpy as np
from data_sql import *
from sklearn.ensemble import GradientBoostingRegressor


def learn_by_gbdt(train_id, train_feat, test_feat):
    gbdt = GradientBoostingRegressor(
        loss='ls'
        , learning_rate=0.1
        , n_estimators=100
        , subsample=1
        , min_samples_split=2
        , min_samples_leaf=1
        , max_depth=3
        , init=None
        , random_state=None
        , max_features=None
        , alpha=0.9
        , verbose=0
        , max_leaf_nodes=None
        , warm_start=False
    )

    # train_feat=np.genfromtxt("train_feat.txt",dtype=np.float32)
    # train_id=np.genfromtxt("train_id.txt",dtype=np.float32)
    # test_feat=np.genfromtxt("test_feat.txt",dtype=np.float32)
    # test_id=np.genfromtxt("test_id.txt",dtype=np.float32)
    gbdt.fit(train_feat, train_id)
    pred = gbdt.predict(test_feat)
    return pred


t_data = tianchi_data()

# query for all artists
sql = 'SELECT distinct(artist_id) FROM music_tianchi.mars_tianchi_songs;'
arr_artist = t_data.query(sql)
predict_data = []

# date array
arr_pri_date = []
d1 = datetime.datetime(2015, 3, 1)
for i in range(0, 183):
    d = d1 + datetime.timedelta(days=i)
    arr_pri_date.append(d.strftime('%Y%m%d'))

arr_date = []
d1 = datetime.datetime(2015, 9, 1)
for i in range(0, 60):
    d = d1 + datetime.timedelta(days=i)
    arr_date.append(d.strftime('%Y%m%d'))

v = 0
for artist_item in arr_artist:
    artist_id = artist_item[0]
    sql = "SELECT count(*),Ds as play_time " + \
          'FROM music_tianchi.mars_tianchi_user_actions,music_tianchi.mars_tianchi_songs ' + \
          "WHERE mars_tianchi_user_actions.song_id=mars_tianchi_songs.song_id AND artist_id = '" + \
          artist_id + "' group by Ds;"
    print 'querying ', v, ' art ' + artist_id
    v += 1
    arr_ret = t_data.query(sql)
    arr_play_time = [0] * 183
    for sub in arr_ret:
        pos = arr_pri_date.index(str(sub[1]))
        arr_play_time[pos] = sub[0]
    xarr = []
    yarr = []
    ytarr = []
    for i in range(0, 183):
        xarr.append([i])
        yarr.append(arr_play_time[i])
        # ytarr.append([i+60])
    for i in range(0, 60):
        ytarr.append([i+183])
    x = np.array(xarr, dtype=np.float32)
    y = np.array(yarr, dtype=np.float32)

    x_test = np.array(range(183, 183 + 60), dtype=np.float32)
    predict_y = learn_by_gbdt(y, x, ytarr)
    print x
    print y
    print predict_y
    for i in range(0, 60):
        predict_data.append((artist_id, int(round(predict_y[i])), arr_date[i]))

csvfile = file("csv_gbdt525.csv", 'wb')
writer = csv.writer(csvfile)

writer.writerows(predict_data)

csvfile.close()
