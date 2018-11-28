from helpers import data_loader as dl
import numpy as np
from lstm_optimizer import do_optimize
from helpers.email_notifier import notify
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))
import math


save_data = True
load_data = False

def optAndNotify(data, labels):
    # try:
    do_optimize(2, data, labels)
    # finally:
    #     notify("volume model")

if load_data:
    data = np.load("processedX.npz")['arr_0']
    labels = np.load("processedY.npz")['arr_0']
    optAndNotify(data, labels)
    quit()

for n in [1]:
# for n in [25, 30, 35, 40, 50]:
# for n in [60, 70, 80, 90, 100]:
    filename = 'Data/aggregatedOutput' + str(n) + 'k.csv'
    csv_rows = dl.load_csv(filename)
    # csv_rows = dl.load_csv('Data/Candles1HDec2014-May2018.csv')
    print('loading', filename)
    headers = csv_rows.pop(0)

    percentized = []
    raw = []
    last_row = None
    labels = []
    # %ize the prices
    for row in reversed(csv_rows): # oldest to newest
        if last_row == None:
            last_row = row
            continue
        pct_row = []
        raw_row = []
        for i in range(2, 6):
            pct_row.append(float(row[i])/float(last_row[i]))
            raw_row.append(float(row[i]))
        pct_row.append(float(row[6]))
        pct_row.append(0.0)  # for hv
        pct_row.append(0.0)  # for ma
        pct_row.append(0.0)  # for macd
        percentized.append(pct_row)

        # create labels
        if row[5] > last_row[5]:
            labels.append(1)
        else:
            labels.append(0)
        last_row = row

        # save raw and working
        raw.append(raw_row)

    # np them
    percentized = np.asarray(percentized)
    # percentized = np.min(percentized, -1)
    labels = np.asarray(labels)
    raw = np.asarray(raw)

    # for hyperparam in range (1,10):
    for hyperparam in [1]:

        look_back = 5 * hyperparam
        n_percentized = len(percentized)
        for i in range(n_percentized-1, -1, -1):
            # standardize the percents
            pct_mean = np.mean(percentized[i-look_back:i, :4])
            pct_std = np.std(percentized[i-look_back:i, :4])
            if pct_std == 0 or math.isnan(pct_std):
                percentized[i, :4] = 0
                percentized[i, 4] = 0
                continue
            percentized[i, :4] = (percentized[i, :4] - pct_mean) / (pct_std*3)

            # standardize the log volume
            vol = percentized[i, 4]
            mean = np.mean(percentized[i-look_back:i, 4])
            std = np.std(percentized[i-look_back:i, 4])
            std_vol = (vol - mean) / (std*3)
            percentized[i, 4] = std_vol

            # add historical volatility
            log_std = np.log(np.std(raw[i-look_back:i, 3]))
            hv = log_std / 3
            hv = min(hv, 1)
            hv = max(hv, -1)
            percentized[i, 5] = hv

        look_back12 = 12
        look_back26 = 26
        ma_diff_working = np.zeros(n_percentized)
        macd_working = np.zeros(n_percentized)
        for i in range(0, n_percentized):
            # add ma
            first_ind = 0
            if i > look_back:
                first_ind = i - look_back
            ma = np.average(raw[first_ind:i, 3])
            ma_diff_working[i] = raw[i, 3] - ma
            ma_diff_mean = np.mean(ma_diff_working[first_ind:i])
            ma_diff_std = np.std(ma_diff_working[first_ind:i])
            percentized[i, 6] = (ma_diff_working[i] - ma_diff_mean) / (ma_diff_std * 3)

            # add macd
            if i < look_back12:
                ma12 = np.average(raw[0:i, 3])
            if i < look_back26:
                ma26 = np.average(raw[0:i, 3])
            if i >= look_back12:
                ma12 = np.average(raw[i - look_back12:i, 3])
            if i >= look_back26:
                ma26 = np.average(raw[i - look_back26:i, 3])
            if i == 0:
                ma12 = 0.0
                ma26 = 0.0
            macd_working[i] = ma12 - ma26
            macd_mean = np.mean(macd_working[first_ind:i])
            macd_std = np.std(macd_working[first_ind:i])
            percentized[i, 7] = (macd_working[i] - macd_mean) / (macd_std*3)

        # make it predictive
        labels = labels[1:]
        percentized = percentized[:-1]

        # remove the first 26 rows that don't have the correct std because there wasn't enough historical data
        first_index = max(look_back26, look_back)
        labels = labels[first_index - 1:]
        percentized = percentized[first_index - 1:]

        for i in range(0, 8):
            print(np.std(percentized[:, i]))

        # batch it
        time_steps = 2 ** 5
        data_samples = []
        label_samples = []
        num_time_pts = len(labels)
        for i in range(0, num_time_pts):
            sample_data = []
            for j in range(0, time_steps):
                if i + j >= num_time_pts:
                    break
                sample_data.append(percentized[i + j])
            if len(sample_data) == time_steps:
                data_samples.append(sample_data)
                label_samples.append(labels[i + j])

        data_samples = np.asarray(data_samples, dtype='float16')
        label_samples = np.asarray(label_samples, dtype='float16')

        if save_data:
            np.savez("processedX.npz", data_samples)
            np.savez("processedY.npz", label_samples)

        print('xy hyperparam', hyperparam)
        optAndNotify(data_samples, label_samples)
