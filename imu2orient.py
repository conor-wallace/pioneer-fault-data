#filter out all columns that we don't need for our RNN
#also filter out time stamps that we don't need
#data we want: time stamp, orientation.x, orientation.y
#appended data: velocity.x, tire_class

'''
tire_class[0] = all full
tire_class[1] = right flat
tire_class[2] = left flat
'''

import csv

with open('imu_data.csv', mode='r') as imu_csvfile, open('training_data.csv', mode='a') as train_csvfile:
    imu_csv_reader = csv.reader(imu_csvfile, delimiter=',')
    train_csv_writer = csv.writer(train_csvfile, delimiter=',')
    line_count = 0
    x_error = y_error = 0
    for row in imu_csv_reader:
        if line_count == 0:
            print("Omitting Header.")
            line_count += 1
        elif line_count == 1:
            x_error += float(row[4])
            y_error += float(row[5])
            line_count += 1
        else:
            if float(row[4])-x_error == 0.0 and float(row[5])-y_error == 0.0:
                print("Pioneer has not moved yet. Omitting data.")
            else:
                train_csv_writer.writerow([int(row[1]),float(row[4])-x_error,float(row[5])-y_error,'0.2','2'])
                line_count += 1

    line_count -= 2
    print('%d data points collected.' % line_count)
