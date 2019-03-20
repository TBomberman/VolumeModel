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

look_back12 = 12
look_back26 = 26

def optAndNotify(data, labels):
    # try:
    do_optimize(2, data, labels)
    # finally:
    #     notify("volume model")


def get_label(current_candle, next_candle):
    if next_candle[5] > current_candle[5]:
        return 1
    else:
        return 0


def get_percentized_row(history):
    pct_rows = []
    for i in range(1, len(history)):
        last_row = history[i-1]
        curr_row = history[i]
        pct_row = []
        for j in range(2, 6):
            pct_row.append(float(curr_row[j]) / float(last_row[j]))
        pct_rows.append(pct_row)
    pct_rows = np.asarray(pct_rows)
    pct_mean = np.mean(pct_rows)
    pct_std = np.std(pct_rows)
    return (pct_rows[len(pct_rows)-1] - pct_mean) / (pct_std*3)


def get_float_index_history(list_history, index):
    index_history = []
    last_val = 0
    for i in range(0, len(list_history)):
        last_val = float(list_history[i][index])
        index_history.append(last_val)
    return last_val, index_history


def get_std_vol(history):
    vol, vol_history = get_float_index_history(history, 6)
    mean = np.mean(vol_history)
    std = np.std(vol_history)
    if std == 0:
        return math.inf
    std_vol = (vol - mean) / (std * 3)
    return std_vol


def get_hv(history):
    dummy, close_history = get_float_index_history(history, 5)
    log_std = np.log(np.std(close_history))
    hv = log_std / 3
    hv = min(hv, 1)
    hv = max(hv, -1)
    return hv


def get_ma_diff(ma_history, look_back):
    ma_diff_working = []
    dummy, close_history = get_float_index_history(ma_history, 5)
    n = len(close_history)
    last_diff = 0
    for i in range(n-look_back, n):
        history = close_history[i-look_back+1:i+1]
        ma = np.average(history)
        last_diff = close_history[i] - ma
        ma_diff_working.append(last_diff)
    ma_diff_mean = np.mean(ma_diff_working)
    ma_diff_std = np.std(ma_diff_working)
    return (last_diff - ma_diff_mean) / (ma_diff_std * 3)


def get_macd(history, stats_look_back):
    n = len(history)
    macd_working = np.zeros(n)
    dummy, close_history = get_float_index_history(history, 5)
    last_macd = 0
    for i in range(1, n+1):
        ma12 = 0.0
        ma26 = 0.0
        if i < look_back12:
            ma12 = np.average(close_history[0:i])
        if i < look_back26:
            ma26 = np.average(close_history[0:i])
        if i >= look_back12:
            ma12 = np.average(close_history[i - look_back12:i])
        if i >= look_back26:
            ma26 = np.average(close_history[i - look_back26:i])

        last_macd = ma12 - ma26
        macd_working[i-1] = last_macd
    macd_mean = np.mean(macd_working[-stats_look_back:])
    macd_std = np.std(macd_working[-stats_look_back:])
    return (last_macd - macd_mean) / (macd_std * 3)


def get_3d_features(history, stats_look_back, batch_size):
    batch_data = []
    n = len(history)
    max_look_back = max(stats_look_back, batch_size)

    for i in range(n-max_look_back+1, n+1):
        batch_row = []
        stats_time_chunk = history[i-stats_look_back-1:i]
        ma_time_chunk = history[i-stats_look_back*2-1:i]
        macd_time_chunk = history[i-stats_look_back-26-1:i]
        pct = get_percentized_row(stats_time_chunk)
        std_vol = get_std_vol(stats_time_chunk[1:])
        hv = get_hv(stats_time_chunk[1:])
        ma = get_ma_diff(ma_time_chunk, stats_look_back)
        macd = get_macd(macd_time_chunk, stats_look_back)
        batch_row.append(pct[0])
        batch_row.append(pct[1])
        batch_row.append(pct[2])
        batch_row.append(pct[3])
        batch_row.append(std_vol)
        batch_row.append(hv)
        batch_row.append(ma)
        batch_row.append(macd)
        batch_data.append(batch_row)
    return batch_data


def process_file(aggregation_count_in_thousands, hyperparam):
    filename = 'Data/aggregatedOutput' + str(aggregation_count_in_thousands) + 'k.csv'
    csv_rows = dl.load_csv(filename)
    print('loading', filename)
    csv_rows.pop(0)
    data = []
    labels = []
    stats_look_back = 5 * hyperparam
    batch_size = 2 ** 5
    history_size = batch_size + max(stats_look_back*2, look_back26 + stats_look_back) + 1

    n_rows = len(csv_rows)

    for i in range(history_size, n_rows - 1):
        batch_time_chunk = csv_rows[i - history_size:i]
        features_3d = get_3d_features(batch_time_chunk, stats_look_back, batch_size)
        data.append(features_3d)
        label = get_label(csv_rows[i], csv_rows[i + 1])
        labels.append(label)

    data = np.asarray(data, dtype='float16')
    labels = np.asarray(labels, dtype='float16')
    for i in range(0, 8):
        print(np.std(data[:, :, i]))
    return data, labels


def main():
    for aggregation_count_in_thousands in [10]:
        for hyperparam in [2]:
            data, labels = process_file(aggregation_count_in_thousands, hyperparam)
            print('xy hyperparam', hyperparam)
            optAndNotify(data, labels)
    return


main()
