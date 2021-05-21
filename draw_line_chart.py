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

# recall_fm, ndcg_fm, precision_fm, hit_ratio_fm = [], [], [], []

recall_fm = load_variavle('recall.txt')
ndcg_fm = load_variavle('ndcg.txt')
precision_fm = load_variavle('precision.txt')
hit_ratio_fm = load_variavle('hit_ratio.txt')

recall_fm = [i.tolist() for i in recall_fm]
ndcg_fm = [i.tolist() for i in ndcg_fm]
precision_fm = [i.tolist() for i in precision_fm]
hit_ratio_fm = [i.tolist() for i in hit_ratio_fm]

# recall_fm = [i[0] for i in recall_fm]
# ndcg_fm = [i[0] for i in ndcg_fm]
# precision_fm = [i[0] for i in precision_fm]
# hit_ratio_fm = [i[0] for i in hit_ratio_fm]

res_20 = [recall_fm[21][0], ndcg_fm[21][0], precision_fm[21][0], hit_ratio_fm[21][0]]
res_40 = [recall_fm[21][1], ndcg_fm[21][1], precision_fm[21][1], hit_ratio_fm[21][1]]
res_60 = [recall_fm[21][2], ndcg_fm[21][2], precision_fm[21][2], hit_ratio_fm[21][2]]
res_80 = [recall_fm[21][3], ndcg_fm[21][3], precision_fm[21][3], hit_ratio_fm[21][3]]
res_100 = [recall_fm[21][4], ndcg_fm[21][4], precision_fm[21][4], hit_ratio_fm[21][4]]

x1 = [1]
x2 = [10*i+9 for i in range(21)]
x = x1 + x2

index = ('recall', 'NDCG', 'precision', 'hit_ratio')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
plt.rcParams['axes.unicode_minus'] = False
plt.title('precision随训练次数的变化')  # 折线图标题

plt.xlabel('评价指标')  # x轴标题
# plt.ylabel('precision')  # y轴标题

bar_width = 0.18  # 条形宽度
index_20 = np.arange(len(index))  # 男生条形图的横坐标
index_40 = index_20 + bar_width  # 女生条形图的横坐标
index_60 = index_40 + bar_width  # 女生条形图的横坐标
index_80 = index_60 + bar_width  # 女生条形图的横坐标
index_100 = index_80 + bar_width  # 女生条形图的横坐标

# 使用两次 bar 函数画出两组条形图
plt.bar(index_20, height=res_20, width=bar_width, color='r', label='top-20')
plt.bar(index_40, height=res_40, width=bar_width, color='g', label='top-40')
plt.bar(index_60, height=res_60, width=bar_width, color='b', label='top-60')
plt.bar(index_80, height=res_80, width=bar_width, color='c', label='top-80')
plt.bar(index_100, height=res_100, width=bar_width, color='y', label='top-100')

plt.legend()  # 显示图例
plt.xticks(index_20 + bar_width*4 / 2, index)  # 让横坐标轴刻度显示 models 里的算法名称， index_recall + bar_width/2 为横坐标轴刻度的位置
# plt.ylabel('购买量')  # 纵坐标轴标题
plt.title('列表长度对评价指标的影响')  # 图形标题

plt.show()

