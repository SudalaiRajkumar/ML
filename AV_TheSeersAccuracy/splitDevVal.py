import csv
from datetime import datetime

with open("../Data/Train_seers_accuracy.csv") as train_file:
        dev_file = open("../Data/dev.csv","w")
        val_file = open("../Data/val.csv","w")

        dev_writer = csv.writer(dev_file)
        val_writer = csv.writer(val_file)

        reader = csv.reader(train_file)
        header = reader.next()
        dev_writer.writerow(header)
        val_writer.writerow(header)
        date_index = header.index("Transaction_Date")

        dev_counter = 0
        val_counter = 0
        total_counter = 0
        for row in reader:
                #print row
                date_val = datetime.strptime(row[date_index], "%d-%b-%y")
                if date_val.year == 2006:
                        val_writer.writerow(row)
                        val_counter += 1
                else:
                        dev_writer.writerow(row)
                        dev_counter += 1
                total_counter += 1
                if total_counter % 10000 == 0:
                        print total_counter, dev_counter, val_counter

        dev_file.close()
        val_file.close()

