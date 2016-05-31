# -*- coding: utf-8 -*-

from data_sql import *


n_steps = 10
total_days = 183
predict_days = 61

def get_orignal_data():
    t = tianchi_data()
    sql = """
    SELECT artist_id,count(if(action_type=1,true,null)) as play_time,count(if(action_type=2,true,null)) as down_time,count(if(action_type=3,true,null)) as favo_time,count(1) as amount,Ds
    FROM music_tianchi.mars_tianchi_user_actions,music_tianchi.mars_tianchi_songs
    WHERE mars_tianchi_user_actions.song_id=mars_tianchi_songs.song_id
    group by artist_id,Ds;
        """
    arr_ret = t.query(sql)

    return arr_ret

def get_train_test(data, split = 0.2):
    train_feat = []
    train_id = []
    test_feat = []
    test_id = []