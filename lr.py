# -*- coding: utf-8 -*-
from data_sql import *
from scipy import
import numpy as np
import csv
import datetime

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
    print 'querying ',v,' art '+ artist_id
    v+= 1
    arr_ret = t_data.query(sql)
    arr_play_time = [0] * 183
    for sub in arr_ret:
        pos = arr_pri_date.index(str(sub[1]))
        arr_play_time[pos] = sub[0]
    x = np.array(range(0, 183))
    y = np.array(arr_play_time)
    slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x, y)
    x_test = np.array(range(183, 183 + 60))
    predict_y = intercept + slope * x_test
    for i in range(0,60):
        predict_data.append((artist_id, int(round(predict_y[i])),arr_date[i]))

csvfile = file("csv_test.csv", 'wb')
writer = csv.writer(csvfile)

writer.writerows(predict_data)

csvfile.close()
