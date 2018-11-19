from helpers import data_loader as dl
import csv

def aggregateByTicks():
    csv_rows = dl.load_csv('Data/ES 12-18.Last.txt')

    aggregated_rows = []
    aggregated_rows.append(['counter', 'datetime', 'low', 'high', 'open', 'close', 'vol'])
    n = 100

    n_aggregated_tioks = 1000 * n

    with open("aggregatedOutput" + str(n) + "k.csv", "w") as file:
        writer = csv.writer(file)

        counter = 0
        low, high, open_, close, vol = 0,0,0,0,0
        for row in csv_rows:
            counter += 1
            tokens = row[0].split(';')
            datetime = tokens[0]
            last = float(tokens[1])
            tick_vol = int(tokens[4])
            if counter%n_aggregated_tioks == 1:
                open_ = last
                low = last
                high = last
                vol = 0
            vol += tick_vol
            if last < low:
                low = last
            if last > high:
                high = last
            if counter%n_aggregated_tioks == 0:
                close = last
                row = [counter, datetime, low, high, open_, close, vol]
                aggregated_rows.append(row)
                if counter % (10*n_aggregated_tioks) == 0:
                    print(row)

        writer.writerows(aggregated_rows)

def aggregateByVolume():
    csv_rows = dl.load_csv('Data/ES 12-18.Last.txt')

    for n in [1, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]:
        aggregated_rows = []
        aggregated_rows.append(['counter', 'datetime', 'low', 'high', 'open', 'close', 'vol'])

        n_aggregated_vol = 1000 * n
        next_open_tick_counter = 1

        with open("aggregatedVolOutput" + str(n) + "k.csv", "w") as file:
            writer = csv.writer(file)

            tick_counter = 0
            low, high, open_, close, vol = 0, 0, 0, 0, 0
            for row in csv_rows:
                tick_counter += 1

                # get values
                tokens = row[0].split(';')
                datetime = tokens[0]
                last = float(tokens[1])
                tick_vol = int(tokens[4])

                # store it in the save variables
                if tick_counter == next_open_tick_counter:
                    open_ = last
                vol += tick_vol
                if last < low:
                    low = last
                if last > high:
                    high = last

                # write candle if overflows
                if vol > n_aggregated_vol:
                    close = last
                    row = [tick_counter, datetime, low, high, open_, close, vol]
                    aggregated_rows.append(row)
                    print(row)

                    # prepare for next candle
                    next_open_tick_counter = tick_counter + 1
                    low = last
                    high = last
                    vol = 0

            writer.writerows(aggregated_rows)

aggregateByTicks()