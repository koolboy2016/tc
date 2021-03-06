import csv
import datetime

from data_sql import *

t_data = tianchi_data()

# query for all artists
sql = 'SELECT distinct(song_id) FROM music_tianchi.mars_tianchi_songs;'
arr_song = t_data.query(sql)
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

t_data.query("TRUNCATE music_tianchi.plays_songs;")

v = 0
p_pg = 0
for song_item in arr_song:
    song_data = []
    song_id = song_item[0]
    sql = "SELECT mars_tianchi_songs.song_id,count(if(action_type=1,true,null)) as play_time,count(if(action_type=2,true,null)) as down_time,count(if(action_type=3,true,null)) as favo_time,Gender,Ds " \
          "FROM music_tianchi.mars_tianchi_user_actions,music_tianchi.mars_tianchi_songs " \
          "WHERE mars_tianchi_user_actions.song_id=mars_tianchi_songs.song_id and mars_tianchi_songs.song_id='" + song_id + "' " \
                                                                                                                            "group by Ds"
    v += 1
    c_pg = v*100/len(arr_song)
    if p_pg != c_pg:
        print 'progressing ',c_pg
        p_pg = c_pg
    arr_ret = t_data.query(sql)
    arr_play_time = [0] * 183
    arr_down_time = [0] * 183
    arr_favor_time = [0] * 183
    for sub in arr_ret:
        pos = arr_pri_date.index(str(sub[5]))
        arr_play_time[pos] = sub[1]
        pos = arr_pri_date.index(str(sub[5]))
        arr_down_time[pos] = sub[2]
        pos = arr_pri_date.index(str(sub[5]))
        arr_favor_time[pos] = sub[3]

    sql = "insert into plays_songs (song_id,play_time,down_time,favor_time,Ds) values(%s,%s,%s,%s,%s)"
    for i in range(0, 183):
        song_data.append(
            (song_id, (arr_play_time[i]), (arr_down_time[i]), (arr_favor_time[i]),arr_pri_date[i]))
    t_data.insert_many(sql, song_data)
