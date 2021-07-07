import numpy as np

rating_file = '../data/' + 'last-fm-small' + '/ratings_final'
rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
a=[]
for item in rating_np:
    if item[2] == 1:
        a.append(item)
b=np.array(a)
np.save('../data/' + 'last-fm-small' + '/rating_pos' + '.npy', rating_np)
