import csv

with open('../config/results.csv', "rb") as file_in:
    reader = csv.reader(file_in)
    with open('fuzzyResults.csv', "wb") as file_out:
        writer = csv.writer(file_out)
        line_count = 0
        for row in reader:
            writer.writerow(row[1:])
