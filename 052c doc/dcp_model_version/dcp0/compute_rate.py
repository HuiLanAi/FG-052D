import os

rate = []
dim = [64*64, 64*64, 64*64, 64*64, 64*128, 128*128, 128*128, 128*256, 256*256, 256*256, 60*256]
f = open("droprate.txt", "r")
for data in f.realines():
    rate.append(data)
f.close()