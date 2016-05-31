# -*- coding: utf-8 -*-

from data_sql import *
import numpy as np
import pandas as pd

n_steps = 10
total_days = 183
predict_days = 61


def get_mean_std_diff(arr_pt, arr_dt, arr_ft):
    avg_plays = arr_pt.mean()
    avg_down = arr_dt.mean()
    avg_favor = arr_ft.mean()

    std_plays = arr_pt.std()
    std_down = arr_dt.std()
    std_favor = arr_ft.std()

    avg_plays_diff = np.diff(arr_pt).mean()
    avg_down_diff = np.diff(arr_pt).mean()
    avg_favor_diff = np.diff(arr_pt).mean()

    std_plays_diff = np.diff(arr_pt).std()
    std_down_diff = np.diff(arr_pt).std()
    std_favor_diff = np.diff(arr_pt).std()
    return [avg_plays, avg_down, avg_favor, std_plays, std_down, std_favor, avg_plays_diff, avg_down_diff,
            avg_favor_diff, std_plays_diff, std_down_diff, std_favor_diff]


if __name__ == '__main__':
    tianchi = tianchi_data()
    sql = 'SELECT distinct(artist_id) FROM music_tianchi.mars_tianchi_songs;'
    arr_artist = tianchi.query(sql)
    predict_data = []
    for artist_item in arr_artist:
        artist_id = artist_item[0]
        sql = "SELECT play_time,down_time,favor_time,gender,day_of_begin,day_of_week,weekend FROM music_tianchi.plays WHERE artist_id = '" + artist_id + "' ;"
        arr_data = pd.DataFrame(np.array(tianchi.query(sql)))
        train_feat = []
        train_id = []
        predict_feat = []
        for sp in range(0, total_days - n_steps):
            arr_pt = np.array(arr_data.iloc[sp:(sp + n_steps), 0].as_matrix())
            arr_dt = np.array(arr_data.iloc[sp:(sp + n_steps), 1].as_matrix())
            arr_ft = np.array(arr_data.iloc[sp:(sp + n_steps), 2].as_matrix())
            feat_for_time = get_mean_std_diff(arr_pt, arr_dt, arr_ft)
            feat_for_normal = np.array(arr_data.iloc[(sp + n_steps)].as_matrix())
            feat = np.vstack((feat_for_time,feat_for_normal))
            tid = arr_data.ix[(sp + n_steps +1, 0)]
            train_feat.append(feat)
            train_id.append(tid)
            print feat,' ',tid
