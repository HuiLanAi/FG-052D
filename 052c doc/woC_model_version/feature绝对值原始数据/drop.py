import os
ratio = 0.90
file_name = "absmean9.txt"

data = []
drop_index = []

f = open(file_name, "r")
for num in f.readlines():
    data.append(float(num))

drop_num = (int)(ratio * len(data))

for i in range (drop_num):
    index = data.index(min(data))
    drop_index.append(index)
    data[index] = 1000

for i in range (drop_num):
    print(drop_index[i])
