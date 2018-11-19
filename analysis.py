from helpers import data_loader as dl
from functools import reduce

for n in [1, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]:
    filename = 'Data/aggregatedOutput' + str(n) + 'k.csv'
    csv_rows = dl.load_csv(filename)
    print('loading', filename)
    headers = csv_rows.pop(0)

    last_time = 0
    min_time_span = 100000
    max_time_span = 0
    spans = []
    counter = 0
    for row in csv_rows:
        tokens = row[1].split(' ')
        time = int(tokens[1])
        counter = int(row[0])
        if last_time == 0 or time < last_time:
            last_time = time
            continue
        time_span = time - last_time
        last_time = time
        spans.append(time_span)
        if time_span < min_time_span:
            min_time_span = time_span
        if time_span > max_time_span:
            max_time_span = time_span
    avg = reduce(lambda x, y: x + y, spans) / len(spans)
    print('1000s', n, 'min', min_time_span, 'max', max_time_span, 'avg', avg)