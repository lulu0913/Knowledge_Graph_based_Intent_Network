import matplotlib.pyplot as plt
import numpy as np

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 输入统计数据
models = ('KGIN', 'MF', 'CKE', 'KGAT', 'KGNN-LS', 'CKAN', 'R-GCN')
recall_alibaba = [0.1152, 0.1095, 0.1103, 0.1030, 0.1039, 0.0970, 0.0860]
ndcg_alibaba = [0.0722, 0.0670, 0.0676, 0.0627, 0.0557, 0.0509, 0.0515]

recall_lastfm = [0.0972, 0.0724, 0.0732, 0.0873, 0.0880, 0.0812, 0.0743]
ndcg_lastfm = [0.0845, 0.0617, 0.0630, 0.0744, 0.0642, 0.0660, 0.0631]

bar_width = 0.3  # 条形宽度
index_recall = np.arange(len(models))  # 男生条形图的横坐标
index_ndcg = index_recall + bar_width  # 女生条形图的横坐标

# 使用两次 bar 函数画出两组条形图
plt.bar(index_recall, height=recall_lastfm, width=bar_width, color='r', label='recall')
plt.bar(index_ndcg, height=ndcg_lastfm, width=bar_width, color='g', label='ndcg')

plt.legend()  # 显示图例
plt.xticks(index_recall + bar_width / 2, models)  # 让横坐标轴刻度显示 models 里的算法名称， index_recall + bar_width/2 为横坐标轴刻度的位置
# plt.ylabel('购买量')  # 纵坐标轴标题
plt.title('Alibaba-iFashion 数据集推荐结果评估')  # 图形标题

plt.show()