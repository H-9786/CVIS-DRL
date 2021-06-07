import matplotlib.pyplot as plt

figsize = 14, 5
plt.subplots(figsize=figsize)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.15)
plt.rcParams['figure.figsize'] = (11, 4)

plt.subplot(1,2,1)
X = ['Fixed-time', 'Longest-queue-first', 'DQN', 'TD3-one-agent', 'TD3-two-agent', 'TD3-three-agent']
Y = [74.39, 73.13, 69.02, 69.87, 67.26, 57.36]
plt.bar(X, Y, 0.4, color="steelblue", hatch='', edgecolor='black')
plt.xlabel("")
plt.ylabel("Average fuel consumption per veh / ml")

for a, b in zip(X, Y):
 plt.text(a, b, '%.2f' % b, ha='center', va='bottom')
plt.xticks(rotation = 10)

plt.grid(axis="y", linestyle='-.')
plt.ylim(0, 79)

plt.subplot(1,2,2)
X = ['Fixed-time', 'Longest-queue-first', 'DQN', 'TD3-one-agent', 'TD3-two-agent', 'TD3-three-agent']
Y = [173.04, 170.13, 160.57, 162.55, 156.46, 133.44]
plt.bar(X, Y, 0.4, color="cornflowerblue", hatch='', edgecolor='black')
plt.xlabel("")
plt.ylabel("Average total CO2 emission per veh / g")

for a, b in zip(X, Y):
 plt.text(a, b, '%.2f' % b, ha='center', va='bottom')
plt.xticks(rotation = 10)
plt.grid(axis="y", linestyle='-.')
plt.ylim(0, 185)
#plt.show()
plt.savefig('co2_fuel_consumption.pdf')

# import matplotlib.pyplot as plt
# import numpy as np
#
# # 输入统计数据
# X = ('Fixed-time', 'Longest-queue-first', 'DQN', 'TD3-one-agent', 'TD3-two-agent', 'TD3-three-agent')
# Y1 = [74.39, 73.13, 69.02, 69.87, 67.26, 57.36]
# Y2 = [17.30, 17.01, 16.06, 16.25, 15.65, 13.34]
#
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# bar_width = 0.3  # 条形宽度
# index_male = np.arange(len(X))  # 男生条形图的横坐标
# index_female = index_male + bar_width  # 女生条形图的横坐标
#
# # 使用两次 bar 函数画出两组条形图
# ax1.bar(index_male, height=Y1, width=bar_width, color='b', label='1')
# ax2.bar(index_female, height=Y2, width=bar_width, color='g', label='2')
#
# plt.legend()  # 显示图例
# plt.xticks(index_male + bar_width/2, X)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
# # plt.ylabel('购买量')  # 纵坐标轴标题
# # plt.title('购买饮用水情况的调查结果')  # 图形标题
# plt.xticks(rotation = 10)
# plt.show()
