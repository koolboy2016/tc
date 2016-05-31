# -*- coding: utf-8 -*-

from data_sql import *
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import csv
import datetime,time

n_steps = 10
total_days = 183
predict_days = 61


def train_by_gbdt(train_feat, train_id):
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
    print train_feat
    print train_id
    gbdt.fit(train_feat, train_id)
    # pred = gbdt.predict(test_feat)
    return gbdt


def get_mean_std_diff(arr_pt, arr_dt, arr_ft):
    avg_plays = arr_pt.mean()
    avg_down = arr_dt.mean()
    avg_favor = arr_ft.mean()

    std_plays = arr_pt.std()
    std_down = arr_dt.std()
    std_favor = arr_ft.std()

    avg_plays_diff = np.diff(arr_pt).mean()
    avg_down_diff = np.diff(arr_dt).mean()
    avg_favor_diff = np.diff(arr_ft).mean()

    std_plays_diff = np.diff(arr_pt).std()
    std_down_diff = np.diff(arr_dt).std()
    std_favor_diff = np.diff(arr_ft).std()
    return [avg_plays, avg_down, avg_favor, std_plays, std_down, std_favor, avg_plays_diff, avg_down_diff,
            avg_favor_diff, std_plays_diff, std_down_diff, std_favor_diff]


def get_predict_date():
    arr_date = []
    d1 = datetime.datetime(2015, 9, 1)
    for i in range(0, 60):
        d = d1 + datetime.timedelta(days=i)
        arr_date.append(d.strftime('%Y%m%d'))
    return arr_date


if __name__ == '__main__':
    tianchi = tianchi_data()
    predict_date = get_predict_date()
    sql = 'SELECT distinct(artist_id) FROM music_tianchi.mars_tianchi_songs;'
    arr_artist = tianchi.query(sql)
    predict_data = []
    for artist_item in arr_artist:
        artist_id = artist_item[0]
        sql = "SELECT play_time,down_time,favor_time,gender,day_of_begin,day_of_week,weekend FROM music_tianchi.plays WHERE artist_id = '" + artist_id + "' order by day_of_begin;"
        arr_data = pd.DataFrame(np.array(tianchi.query(sql)))
        train_feat = []
        train_idp = []
        train_idd = []
        train_idf = []

        predict_feat = []
        predict_id = []
        gender = arr_data.ix[0,3]
        for sp in range(0, total_days - n_steps - 1):
            arr_pt = np.array(arr_data.iloc[sp:(sp + n_steps), 0].as_matrix())
            arr_dt = np.array(arr_data.iloc[sp:(sp + n_steps), 1].as_matrix())
            arr_ft = np.array(arr_data.iloc[sp:(sp + n_steps), 2].as_matrix())
            feat_for_time = get_mean_std_diff(arr_pt, arr_dt, arr_ft)
            feat_for_normal = np.array(arr_data.iloc[(sp + n_steps)].as_matrix())
            feat = np.concatenate((feat_for_time, feat_for_normal))

            if sp != total_days - n_steps - 2:
                tid = (arr_data.iloc[sp + n_steps + 1, 0:3].as_matrix())
                train_feat.append(feat)

                train_idp.append(tid[0])
                train_idd.append(tid[1])
                train_idf.append(tid[2])
                # print feat, ' ', tid
            else:
                predict_feat.append(feat)
        model_play = train_by_gbdt(train_feat, train_idp)
        model_down = train_by_gbdt(train_feat, train_idd)
        model_favor = train_by_gbdt(train_feat, train_idf)

        for spt in range(0, predict_days):
            pred_p = model_play.predict(predict_feat)
            pred_d = model_down.predict(predict_feat)
            pred_f = model_favor.predict(predict_feat)

            predict_id.append(pred_p)
            print pred_p,pred_d,pred_f

            wkd = 0
            dow = int(predict_feat[len(predict_feat)-2])%7
            if dow in [5,6]:
                wkd = 1
            next_row = np.array([pred_p,pred_d,pred_f,predict_feat[-4], predict_feat[-3]+1, dow, wkd])
            res = arr_data.append(next_row, ignore_index=True)

            arr_pt = np.array(arr_data.iloc[-n_steps:, 0].as_matrix())
            arr_dt = np.array(arr_data.iloc[-n_steps:, 1].as_matrix())
            arr_ft = np.array(arr_data.iloc[-n_steps:, 2].as_matrix())
            feat_for_time = get_mean_std_diff(arr_pt, arr_dt, arr_ft)
            feat_for_normal = np.array(arr_data.iloc[-1].as_matrix())
            predict_feat = np.concatenate((feat_for_time, feat_for_normal))
        time.sleep(1000)



