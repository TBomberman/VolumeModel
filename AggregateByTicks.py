from helpers import data_loader as dl
import csv

csv_rows = dl.load_csv('Data/ES 12-18.Last.txt')

aggregated_rows = []
aggregated_rows.append(['counter', 'datetime', 'low', 'high', 'open', 'close', 'vol'])

with open("aggregatedOutput.csv", "w") as f:
    writer = csv.writer(f)

    counter = 0
    low, high, open, close, vol = 0,0,0,0,0
    for row in csv_rows:
        counter += 1
        tokens = row[0].split(';')
        datetime = tokens[0]
        last = float(tokens[1])
        tick_vol = int(tokens[4])
        if counter%100 == 1:
            open = last
            low = last
            high = last
            vol = 0
        vol += tick_vol
        if last < low:
            low = last
        if last > high:
            high = last
        if counter%100 == 0:
            close = last
            row = [counter, datetime, low, high, open, close, vol]
            aggregated_rows.append(row)
            if counter % 10000 == 0:
                print(row)

    writer.writerows(aggregated_rows)