import os

rate = []
conv1_dim = [64*3, 64*64, 64*64, 64*64, 64*128, 128*128, 128*128, 128*256, 256*256, 256*256, 60*256]
conv9_dim = [9*64*64, 9*64*64, 9*64*64, 9*64*64, 9*64*128, 9*128*128, 9*128*128, 9*128*256, 9*256*256, 9*256*256]

rate_1 = [0, 0.4, 0.25, 0.2, 0.15, 0.52, 0.42, 0.50, 0.95, 0.92, 0]
# rate_9 = [0.4, 0.25, 0.2, 0.15, 0.52, 0.42, 0.50, 0.95, 0.92, 0]
rate_9 = [0.6, 0.4, 0.35, 0.35, 0.7, 0.62, 0.7, 0.95, 0.92, 0]
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

drop_9 += (sum(conv9_dim[:])-drop_9) * 0.8


rate = (drop_1 + drop_9) / total
ratio = 1 / (1-rate) 


skip_graph_rate = drop_1 / sum(conv1_dim[:-1])


print(rate)
print(ratio)
print(skip_graph_rate)


fea_dim = [25*150*3, 25*150*64, 25*150*64, 25*150*64, 25*75*128, 25*75*128, 25*75*128, 25*38*256, 25*38*256, 25*38*256]
graph_ops = 0
for i in range (10):
    graph_ops += fea_dim[i] * 25 * 3
graph_ops *= 2

# graph_ops *= skip_graph_rate * 2

conv1_filter_dim = [3, 64, 64, 64, 128, 128, 128, 256, 256, 256]
conv1_ops = 0
for i in range (10):
    conv1_ops += fea_dim[i] * conv1_filter_dim[i]
conv1_ops += sum(fea_dim[:]) * 2
# conv1_ops *= skip_graph_rate


conv9_filter_dim = [64, 64, 64, 64, 128, 128, 128, 256, 256, 256]
conv9_ops = 0
for i in range (10):
    conv9_ops += fea_dim[i] * conv9_filter_dim[i] * 9
conv9_ops += sum(fea_dim[:]) * 2
conv9_ops *= (drop_9 / sum(conv9_dim[:-1]))


print(graph_ops+conv1_ops+conv9_ops)

print(graph_ops / (graph_ops + conv1_ops))
