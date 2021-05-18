import matplotlib.pyplot as plt

recall_alibaba, ndcg_alibaba, precision_alibaba, hit_ratio_alibaba = [], [], [], []
recall_fm, ndcg_fm, precision_fm, hit_ratio_fm = [], [], [], []
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

x1 = [1]
x2 = [10*i+9 for i in range(21)]
x = x1 + x2
y1 = [87, 174, 225, 254]
y2 = [24, 97, 202, 225]
y3 = [110, 138, 177, 205]
y4 = [95, 68, 83, 105]

plt.title('扩散速度')  # 折线图标题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
plt.xlabel('时间')  # x轴标题
plt.ylabel('差值')  # y轴标题
plt.plot(x, y1, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
plt.plot(x, y2, marker='o', markersize=3)
plt.plot(x, y3, marker='o', markersize=3)
plt.plot(x, y4, marker='o', markersize=3)

for a, b in zip(x, y1):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)  # 设置数据标签位置及大小
for a, b in zip(x, y2):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
for a, b in zip(x, y3):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
for a, b in zip(x, y4):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

plt.legend(['方案一', '方案二', '方案三', '方案四'])  # 设置折线名称

plt.show()  # 显示折线图

