import os

def gen_drop_channel(f_abs_mean, ratio, serial):
    drop_index = []
    f = open(f_abs_mean, "r")
    abs_mean = []
    for num in f.readlines():
        abs_mean.append(float(num))
    f.close()
    drop_num = (int)(ratio * len(abs_mean))

    for i in range (drop_num):
        index = abs_mean.index(min(abs_mean))
        drop_index.append(index)
        abs_mean[index] = 1000
    
    if serial < 10: 
        dst_name = "L" + str(serial) + "_drop.txt"
    else:
        dst_name = "fc_drop.txt"
    
    f = open(dst_name, "w")
    for i in range (len(drop_index)):
        f.write(str(drop_index[i]) + '\n')
    f.close()



def load_ratio(f_ratio):
    ratio = []
    f = open(f_ratio, "r")
    for num in f.readlines():
        ratio.append(float(num))
    f.close()
    return ratio




ratio = load_ratio("droprate.txt")

f_abs_mean = "dcp0_fcmean_abs.txt"
gen_drop_channel(f_abs_mean, ratio[0], 0)