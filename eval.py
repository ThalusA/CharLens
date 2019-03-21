import csv
file_read = open("./emnist/emnist-balanced-train.csv", 'r')
file_write = open("./emnist/emnist_train.csv", 'w', newline='')
reader = csv.reader(file_read, delimiter=',')
writer = csv.writer(file_write, delimiter=',')

def extract():
	for row in reader:
		label = int(row[0])
		if(label <= 35):
			writer.writerow(row)
extract()
