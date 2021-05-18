import matplotlib.pyplot as plt
import pickle
import numpy as np

def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename

def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r

recall_alibaba, ndcg_alibaba, precision_alibaba, hit_ratio_alibaba = [], [], [], []
# recall_fm, ndcg_fm, precision_fm, hit_ratio_fm = [], [], [], []
flag = 0
f = open("./training_log/Alibaba-iFashion.txt", "r", encoding='UTF-8')
line = f.readline()
while line:
    line = f.readline()
    if line != '' and line[0] == '|':
        flag += 1
        flag = flag % 2
        if flag == 0:
            # print('yes\t', line)
            line_res = line.split('|')
            recall_res = line_res[5][2:-2].split(' ')
            ndcg_res = line_res[6][2:-2].split(' ')
            precision_res = line_res[7][2:-2].split(' ')
            hit_ratio_res = line_res[8][2:-2].split(' ')

            recall_alibaba.append(float(recall_res[0]))
            ndcg_alibaba.append(float(ndcg_res[0]))
            precision_alibaba.append(float(precision_res[0]))
            hit_ratio_alibaba.append(float(hit_ratio_res[0]))

f.close()

recall_fm = load_variavle('recall.txt')
ndcg_fm = load_variavle('ndcg.txt')
precision_fm = load_variavle('precision.txt')
hit_ratio_fm = load_variavle('hit_ratio.txt')

recall_fm = [i.tolist() for i in recall_fm]
ndcg_fm = [i.tolist() for i in ndcg_fm]
precision_fm = [i.tolist() for i in precision_fm]
hit_ratio_fm = [i.tolist() for i in hit_ratio_fm]

recall_fm = [i[0] for i in recall_fm]
ndcg_fm = [i[0] for i in ndcg_fm]
precision_fm = [i[0] for i in precision_fm]
hit_ratio_fm = [i[0] for i in hit_ratio_fm]

x1 = [1]
x2 = [10*i+9 for i in range(21)]
x = x1 + x2

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
plt.title('precision随训练次数的变化')  # 折线图标题

plt.xlabel('epochs')  # x轴标题
plt.ylabel('precision')  # y轴标题

plt.plot(x, precision_alibaba, label='Alibaba-iFashion')
plt.plot(x, precision_fm, label='Last-FM')
plt.legend()
# plt.legend(['方案一', '方案二', '方案三', '方案四'])  # 设置折线名称

plt.show()  # 显示折线图

