import pandas as pd
from random import random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

from data_sql import *
import time

max_length = 7

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
    in_dim = 1
    out_dim = 1
    model = Sequential()
    model.add(LSTM(500, return_sequences=True, input_shape=(max_sequence_length, in_dim)))
    model.add(LSTM(500, return_sequences=True))
    model.add(LSTM(500, return_sequences=True))
    model.add(LSTM(500, return_sequences=False))
    model.add(Dense(out_dim, activation='linear'))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    model.fit(X_train, y_train, batch_size=450, nb_epoch=1000, validation_split=0.3)
    predicted = model.predict(X_test)
    return predicted

t_data = tianchi_data()

# query for all artists
sql = 'SELECT distinct(artist_id) FROM music_tianchi.plays;'
arr_artist = t_data.query(sql)
for artist_item in arr_artist:
    print 'handling ',artist_item[0]
    sql = "SELECT plays FROM music_tianchi.plays WHERE artist_id='" + artist_item[0] + "' Order by Ds;"
    p = t_data.query(sql)
    plays = []
    for pi in p:
        plays.append(pi[0])
    plays = pd.DataFrame(plays)
    (X_train, y_train), (X_test, y_test) = train_test_split(plays)  # retrieve data
    # print len(X_test),X_test
    # print len(y_test),y_test
    # time.sleep(1000)
    predicted = train_test_by_lstm(X_train, y_train,X_test)
    rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))
    print rmse


