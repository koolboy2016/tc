# from __future__ import print_function

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import datetime

import statsmodels.api as sm
from data_sql import *

from statsmodels.graphics.api import qqplot
import time

def train_by_arma(dta, predict_date):
    # dta= dta.diff(2)
    # fig = plt.figure(figsize=(12,8))
    # ax1=fig.add_subplot(211)
    # fig = sm.graphics.tsa.plot_acf(dta,lags=40,ax=ax1)
    # ax2 = fig.add_subplot(212)
    # fig = sm.graphics.tsa.plot_pacf(dta,lags=40,ax=ax2)
    # plt.show()
    # time.sleep(1000)

    arma_mod30 = sm.tsa.ARMA(dta, (7, 0)).fit()
    sm.stats.durbin_watson(arma_mod30.resid.values)
    resid = arma_mod30.resid
    stats.normaltest(resid)
    r, q, p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
    predict = arma_mod30.predict('2015m8','2015m10')
    return predict

t_data = tianchi_data()

# query for all artists
sql = 'SELECT distinct(artist_id) FROM music_tianchi.mars_tianchi_songs;'
arr_artist = t_data.query(sql)
for artist_item in arr_artist:
    sql = "SELECT plays FROM music_tianchi.plays WHERE artist_id='" + artist_item[0] + "' Order by Ds;"
    plays = t_data.query(sql)
    pp = []
    for p in plays:
        pp.append(p[0])
    dta = pd.Series(np.array(pp,dtype='float64'))
    dates_train = []
    d1 = datetime.datetime(2015, 3, 1)
    for i in range(0, 183):
        d = d1 + datetime.timedelta(days=i)
        dates_train.append(d)
    dta.index = pd.Index(dates_train)
    # print dta

    dates_test = []
    d1 = datetime.datetime(2015, 9, 1)
    for i in range(0, 60):
        d = d1 + datetime.timedelta(days=i)
        dates_test.append(d)
    predict = train_by_arma(dta,dates_test)
    print predict
