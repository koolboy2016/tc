import csv
import datetime

import numpy as np
import pandas as pd
from keras.layers.core import Dense,Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from data_sql import *
from tc_util import *

max_length = 30
in_dim = 10842
out_dim = 10842
D_batch_size = 100000
D_nb_epoch = 50
D_validation_split = 0.3
rate_of_test = 0.3
predict_date = 61

mod = 'v'


def _load_data(data, n_prev=max_length):
    """
    data should be pd.DataFrame()
    """
    docX, docY = [], []
    for i in range(len(data) - n_prev):
        docX.append(data.iloc[i:i + n_prev].as_matrix())
        docY.append(data.iloc[i + n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY


def train_test_split(df, test_size=0.1):
    """
    This just splits data to training and testing parts
    """
    ntrn = int(round(len(df) * (1 - test_size)))

    X_train, y_train = _load_data(df.iloc[0:ntrn])
    X_test, y_test = _load_data(df.iloc[ntrn:])

    return (X_train, y_train), (X_test, y_test)


def train_test_by_lstm(X_train, y_train, X_test):
    hidden_neurons = 300
    max_sequence_length = max_length

    model = Sequential()
    model.add(LSTM(500, return_sequences=True, input_shape=(max_sequence_length, in_dim)))
    model.add(Dropout(0.2))
    model.add(LSTM(500, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(500, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(500, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(500, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(500, return_sequences=False))
    model.add(Dense(out_dim, activation='linear'))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    model.fit(X_train, y_train, batch_size=D_batch_size, nb_epoch=D_nb_epoch, validation_split=D_validation_split)
    predicted = model.predict(X_test)
    return predicted


def train_by_lstm(X_train, y_train):
    hidden_neurons = 300
    max_sequence_length = max_length

    model = Sequential()
    model.add(LSTM(500, return_sequences=True, input_shape=(max_sequence_length, in_dim)))
    model.add(Dropout(0.2))
    model.add(LSTM(500, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(500, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(500, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(500, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(500, return_sequences=False))
    model.add(Dense(out_dim, activation='linear'))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    model.fit(X_train, y_train, batch_size=D_batch_size, nb_epoch=D_nb_epoch, validation_split=D_validation_split)
    return model


def load_tianchi_data():
    t_data = tianchi_data()
    sql = 'SELECT distinct(song_id) FROM music_tianchi.plays_songs order by song_id;'
    arr_song = t_data.query(sql)
    all_plays = []
    for song_item in arr_song:
        sql = "SELECT play_time FROM music_tianchi.plays_songs WHERE song_id='" + song_item[0] + "' Order by Ds;"
        p = t_data.query(sql)
        plays = []
        for pi in p:
            plays.append(pi[0])
        all_plays.append(plays)
    s_a_map = {}
    a_map = {}
    sql = "SELECT song_id,artist_id FROM music_tianchi.mars_tianchi_songs group by song_id order by song_id;"
    sa_arr = t_data.query(sql)
    for saidx in range(0,len(sa_arr)):
        s_a_map[saidx] = sa_arr[saidx][1]

    sql = "SELECT distinct(artist_id) FROM music_tianchi.mars_tianchi_songs  order by artist_id;"
    sa_arr = t_data.query(sql)
    for saidx in range(0,len(sa_arr)):
        a_map[sa_arr[saidx][0]] = saidx

    playss = pd.DataFrame(all_plays)
    all_plays = playss.T
    return all_plays, arr_song,s_a_map,a_map


def get_score_of_predict(predicted, y_test, data):
    score = 0.0
    for i in range(0, 50):
        thta = 0.0
        fi = 0.0
        for k in range(0, len(y_test)):
            thta = ((predicted[k][i] - y_test[k][i]) / y_test[k][i]) ** 2
            fi += y_test[k][i]
        thta = np.sqrt(thta / len(y_test))
        fi = np.sqrt(fi)
        score += (1 - thta) * fi
    return score

def get_artist_predict(predict_res, s_a_map,a_map):
    n_days = len(predict_res)
    ret = []
    for d in range(0,n_days):
        line = [0]*50
        for i in range(0,len(predict_res)):
            line[a_map[s_a_map[i]]] += predict_res[d][i]
        ret.append(line)
    return ret

data, arr_artist,s_a_map,a_map = load_tianchi_data()
arr_date = []
d1 = datetime.datetime(2015, 9, 1)
for i in range(0, 60):
    d = d1 + datetime.timedelta(days=i)
    arr_date.append(d.strftime('%Y%m%d'))

if mod == 'v':
    # for self-valid
    (X_train, y_train), (X_test, y_test) = train_test_split(data, rate_of_test)

    predicted = train_test_by_lstm(X_train, y_train, X_test)
    artt_p = get_artist_predict(predicted,s_a_map,a_map)
    artt_y = get_artist_predict(y_test,s_a_map,a_map)

    print predicted

    score = get_score_of_predict(artt_p, artt_y, data)
    # print score
    rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))
    print rmse
elif mod == 'c':
    # for contest
    (X_train, y_train), (X_test, y_test) = train_test_split(data, 0)

    model = train_by_lstm(X_train, y_train)
    td = data.iloc[len(data) - max_length:].as_matrix()
    all_predict = []
    predict_data = []
    for k in range(0, predict_date):
        arr = []
        arr.append(td)
        # print np.array(arr)
        # time.sleep(1000)
        # print td
        predicted = model.predict(np.array(arr))[0]
        # predicted = pd.Series(range(0,50))
        # print td
        # print predicted
        # time.sleep(10)
        td = pd.DataFrame(td)
        all_predict.append(predicted)
        # print predicted
        td = td.iloc[1:]
        td.ix[max_length + k] = pd.Series(predicted)
        td = td.iloc[0:].as_matrix()
        if predict_date ==60 :
            for idx in range(len(predicted)):
                predict_data.append((arr_artist[idx][0], int(round(predicted[idx])), arr_date[k]))
        if predict_date==61 and k > 0:
            for idx in range(len(predicted)):
                predict_data.append((arr_artist[idx][0], int(round(predicted[idx])), arr_date[k-1]))
    # print all_predict

    file_name = get_result_name("gbdt_mm")
    csvfile = file(file_name, 'wb')
    writer = csv.writer(csvfile)
    writer.writerows(predict_data)

    csvfile.close()
