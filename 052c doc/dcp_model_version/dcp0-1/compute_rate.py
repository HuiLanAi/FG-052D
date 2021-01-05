import os

rate = []
dim = [64*64, 64*64, 64*64, 64*128, 128*128, 128*128, 128*256, 256*256, 256*256, 60*256]
# dim = [64*64, 64*64, 64*64, 64*128, 128*128, 128*128, 128*256]

f = open("droprate.txt", "r")
for data in f.readlines():
    rate.append((float)(data))
f.close()

drop_n = 0
for i in range (10):
    drop_n += rate[i] * dim[i]

total = sum(dim[:])

print(drop_n / total)