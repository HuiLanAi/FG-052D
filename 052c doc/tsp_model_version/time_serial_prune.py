sample_scheme_list = [
    [1, 3, 5, 7],
    [1, 4, 7],
    [2, 5, 8],
    [1, 6],
    [0, 2, 4, 6, 8],
    [0, 3, 6],
    [0, 5],
    [1, 7]
]

def gen_cavity(sample_scheme_list):
    cavity_list = []
    for i in range (8):
        tmp = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        print(i)
        for j in range(len(sample_scheme_list[i])):
            tmp[sample_scheme_list[i][j]] = 100
        
        while (100 in tmp):
            index = tmp.index(100)
            tmp.pop(index)

        cavity_list.append(tmp)
    return cavity_list

gen_cavity(sample_scheme_list)



