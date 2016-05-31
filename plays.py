import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from data_sql import *
from scipy import stats
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
    sql = "SELECT count(if(action_type=1,true,null)) as play_time,count(if(action_type=2,true,null)) as down_time,count(if(action_type=3,true,null)) as favo_time,Ds " + \
          'FROM music_tianchi.mars_tianchi_user_actions,music_tianchi.mars_tianchi_songs ' + \
          "WHERE mars_tianchi_user_actions.song_id=mars_tianchi_songs.song_id AND artist_id = '" + \
          artist_id + "' group by Ds;"
    print 'querying ', v, ' art ' + artist_id
    v += 1
    arr_ret = t_data.query(sql)
    arr_play_time = [0] * 183
    arr_down_time = [0] * 183
    arr_favor_time = [0] * 183
    for sub in arr_ret:
        pos = arr_pri_date.index(str(sub[3]))
        arr_play_time[pos] = sub[0]

        pos = arr_pri_date.index(str(sub[3]))
        arr_down_time[pos] = sub[1]

        pos = arr_pri_date.index(str(sub[3]))
        arr_favor_time[pos] = sub[2]

    sql = "SELECT distinct Gender FROM music_tianchi.mars_tianchi_songs WHERE artist_id='"+ artist_id +"' LIMIT 0,1 ;"
    g_ret = t_data.query(sql)
    gender = g_ret[0][0]
    for i in range(0, 183):
        predict_data.append((artist_id, int(round(arr_play_time[i])), int(round(arr_down_time[i])), int(round(arr_favor_time[i])),gender, arr_pri_date[i]),i)

csvfile = file("csv_paly.csv", 'wb')
writer = csv.writer(csvfile)

writer.writerows(predict_data)

csvfile.close()
