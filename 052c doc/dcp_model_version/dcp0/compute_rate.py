import os

rate = []
tsp_rate = 0.75
conv1_dim = [64*3*4, 64*64*3, 64*64*3, 64*64*3, 64*128*4, 128*128*4, 128*128*4, 128*256*4, 256*256*4, 256*256*4, 60*256]
conv9_dim = [9*64*64, 9*64*64, 9*64*64, 9*64*64, 9*64*128*2, 9*128*128, 9*128*128, 9*128*256*2, 9*256*256*2, 9*256*256*2]

rate_1 = [0, 0.4, 0.25, 0.2, 0.15, 0.52, 0.42, 0.35, 0.95, 0.92]
rate_9 = [0.4, 0.25, 0.2, 0.15, 0.52, 0.42, 0.35, 0.95, 0.92]
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
for i in range (10):
    drop_1 += rate_1[i] * conv1_dim[i]

drop_9 = 0
for i in range (len(rate_9)):
    drop_9 += rate_9[i] * conv9_dim[i]
drop_9 += (sum(conv9_dim[:])-drop_9)*tsp_rate
# drop_9 += conv9_dim[-1]*0.8


print((drop_1 + drop_9) / total)