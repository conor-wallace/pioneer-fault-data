import csv

with open('new_data.csv', "r") as file_in:
    with open('new_training_set.csv', "a") as file_out:
        writer = csv.writer(file_out)
        line_count = 0
        for row in csv.reader(file_in):
            if line_count == 0:
                pass
            else:
                writer.writerow(row[1:])
            line_count += 1
