import os

rate = []
conv1_dim = [64*3, 64*64, 64*64, 64*64, 64*128, 128*128, 128*128, 128*256, 256*256, 256*256, 60*256]
conv9_dim = [9*64*64, 9*64*64, 9*64*64, 9*64*64, 9*64*128, 9*128*128, 9*128*128, 9*128*256, 9*256*256, 9*256*256]

rate_1 = [0, 0.4, 0.25, 0.2, 0.15, 0.52, 0.42, 0.35, 0.95, 0.92, 0.5]
rate_9 = [0.4, 0.25, 0.2, 0.15, 0.52, 0.42, 0.35, 0.95, 0.92, 0.5]
# dim = [64*64, 64*64, 64*64, 64*128, 128*128, 128*128, 128*256]

# f = open("droprate.txt", "r")
# for data in f.readlines():
#     rate.append((float)(data))
# f.close()

# drop_n = 0
# for i in range (7):
#     drop_n += rate[i] * dim[i]

# total = sum(dim[:])

total = sum(conv1_dim[:])
total += sum(conv9_dim[:])

drop_1 = 0
for i in range (11):
    drop_1 += rate_1[i] * conv1_dim[i]

drop_9 = 0
for i in range (10):
    drop_9 += rate_9[i] * conv9_dim[i]
drop_9 += (sum(conv9_dim[:])-drop_9)/2



print((drop_1 + drop_9) / total)