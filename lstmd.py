import pandas as pd
from random import random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

from data_sql import *
import time
import csv
import datetime

t_start = time.clock()

max_length = 7
in_dim = 1
out_dim = 1
D_batch_size = 5000
D_nb_epoch = 10
D_validation_split = 0.3
rate_of_test = 0.3
predict_date = 61
mod = 'c'

arr_date = []
d1 = datetime.datetime(2015, 9, 1)
for i in range(0, 60):
    d = d1 + datetime.timedelta(days=i)
    arr_date.append(d.strftime('%Y%m%d'))


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


#
# def get_train_test_data(df):
#     """
#     This just splits data to training and testing parts
#     """
#     test_n = 60
#
#     X_train, y_train = _load_data(df.iloc[0:test_n])
#     X_test, y_test = _load_data(df.iloc[test_n:])
#     return (X_train, y_train), (X_test, y_test)


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
    # model.add(LSTM(500, return_sequences=True))
    # model.add(LSTM(500, return_sequences=True))
    # model.add(LSTM(500, return_sequences=True))
    # model.add(LSTM(500, return_sequences=True))
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
    # model.add(LSTM(500, return_sequences=True))
    # model.add(LSTM(500, return_sequences=True))
    # model.add(LSTM(500, return_sequences=True))
    # model.add(LSTM(500, return_sequences=True))
    model.add(LSTM(500, return_sequences=False))
    model.add(Dense(out_dim, activation='linear'))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    model.fit(X_train, y_train, batch_size=D_batch_size, nb_epoch=D_nb_epoch, validation_split=D_validation_split)
    return model


def load_tianchi_data():
    t_data = tianchi_data()
    sql = 'SELECT distinct(artist_id) FROM music_tianchi.plays;'
    arr_artist = t_data.query(sql)
    all_plays = []
    for artist_item in arr_artist:
        sql = "SELECT plays FROM music_tianchi.plays WHERE artist_id='" + artist_item[0] + "' Order by Ds;"
        p = t_data.query(sql)
        plays = []
        for pi in p:
            plays.append(pi[0])
        all_plays.append(plays)

    playss = pd.DataFrame(all_plays)
    all_plays = playss.T
    return all_plays, arr_artist


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


def get_score_of_one_predict(predicted, y_test):
    score = 0.0
    thta = 0.0
    fi = 0.0

    for k in range(0, len(y_test)):
        thta = ((predicted[k][0] - y_test[k][0]) / y_test[k][0]) ** 2
        fi += y_test[k][0]
    thta = np.sqrt(thta / len(y_test))
    fi = np.sqrt(fi)
    score += (1 - thta) * fi
    return score


data, arr_artist = load_tianchi_data()

t_data = tianchi_data()
if mod == 'v':

    sql = 'SELECT distinct(artist_id) FROM music_tianchi.plays;'
    arr_artist = t_data.query(sql)
    score = 0.0
    for artist_item in arr_artist:
        print 'handling ', artist_item[0]
        sql = "SELECT plays FROM music_tianchi.plays WHERE artist_id='" + artist_item[0] + "' Order by Ds;"
        p = t_data.query(sql)
        plays = []
        for pi in p:
            plays.append(pi[0])
        plays = pd.DataFrame(plays)
        (X_train, y_train), (X_test, y_test) = train_test_split(plays, rate_of_test)  # retrieve data
        predicted = train_test_by_lstm(X_train, y_train, X_test)
        a_score = get_score_of_one_predict(predicted, y_test)

        print 'this term, score=', a_score
        score += a_score
    print 'score=', score


elif mod == 'c':
    # for contest
    sql = 'SELECT distinct(artist_id) FROM music_tianchi.plays;'
    arr_artist = t_data.query(sql)
    for artist_item in arr_artist:
        print 'handling ', artist_item[0]
        sql = "SELECT plays FROM music_tianchi.plays WHERE artist_id='" + artist_item[0] + "' Order by Ds;"
        p = t_data.query(sql)
        plays = []
        for pi in p:
            plays.append(pi[0])
        plays = pd.DataFrame(plays)
        (X_train, y_train), (X_test, y_test) = train_test_split(plays, 0)  # retrieve data
        model = train_by_lstm(X_train, y_train)
        td = data.iloc[len(data) - max_length:].as_matrix()
        print td
        time.sleep(1000)
        predict_data = []
        for k in range(0, predict_date):
            arr = []
            arr.append(td)
            predicted = model.predict(np.array(arr))[0]
            td = pd.DataFrame(td)
            td = td.iloc[1:]
            td.ix[max_length + k] = pd.Series(predicted)
            td = td.iloc[0:].as_matrix()
            if predict_date == 60:
                for idx in range(len(predicted)):
                    predict_data.append((arr_artist[idx][0], int(round(predicted[idx])), arr_date[k]))
            if predict_date == 61 and k > 0:
                for idx in range(len(predicted)):
                    predict_data.append((arr_artist[idx][0], int(round(predicted[idx])), arr_date[k - 1]))
        csvfile = file("csv_lstmd1.csv", 'wb')
        writer = csv.writer(csvfile)
        writer.writerows(predict_data)
        csvfile.close()

t_end = time.clock()
print 'elapsed time=', t_end - t_start
