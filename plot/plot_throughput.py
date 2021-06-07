import numpy as np

def text_read(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    x = []

    for line in lines:
        x.append(int(line.split('\n')[0]))

    file.close()
    return x

def add(y):
    x = []
    for i in range(30):
        x.append(sum(y[i*300:i*300+300]))
        #print(x)
    return x

fix_agent = text_read('throughtput_fix.txt')
fix_agent = add(fix_agent)

long_agent = text_read('throughtput_long.txt')
long_agent = add(long_agent)

dqn_agent = text_read('throughtput_dqn.txt')
dqn_agent = add(dqn_agent)

one_agent = text_read('throughtput_td3_one.txt')
one_agent = add(one_agent)

two_agent = text_read('throughtput_td3_two.txt')
two_agent = add(two_agent)

three_agent = text_read('throughtput_td3_three.txt')
three_agent = add(three_agent)

print(sum(fix_agent), sum(long_agent), sum(dqn_agent), sum(one_agent), sum(two_agent), sum(three_agent))


import matplotlib.pyplot as plt

# print(len(h_fixed))
x = np.linspace(0, 9000, 30)

# plt.plot(x, y1, linewidth=2,marker='H', color='steelblue' ,
#          markerfacecolor='powderblue',markersize=7, label=r'$\beta=0.90$')
# plt.plot(x, y2, linewidth=2,color='indianred',marker='o',
#          markerfacecolor='lightcoral' ,markersize=7, label=r'$\beta=0.95$')
# plt.plot(x, y3, linewidth=2, marker='s', color='orange',
#          markerfacecolor='moccasin', markersize=7,label=r'$\beta=0.99$')


plt.plot(x, fix_agent, label='Fixed-time', marker='s', linestyle='dashed', color='orange')
plt.plot(x, long_agent, label='Longest-queue-first', color='indianred',marker='o', linestyle='dashed')
plt.plot(x, dqn_agent, label='DQN', marker='H', linestyle='dashed', color='plum')
plt.plot(x, one_agent, label='TD3-one-agent', marker='^', linestyle='dashed', color='turquoise')
plt.plot(x, two_agent, label='TD3-two-agent', marker='d', linestyle='dashed', color='dodgerblue')
plt.plot(x, three_agent, label='TD3-three-agent', marker='p', linestyle='dashed', color='blueviolet')
# plt.plot(x, w_spatial, label='Spatial-DDPG', marker='h', linestyle='dashed', color='mediumslateblue')
# plt.plot(x, w_our, label='MARL-DSTAN', marker='o',color='steelblue')

plt.ylabel('Number of vehicles leaving the road network / veh', fontsize=10)
plt.xlabel('Time / s', fontsize=12)
plt.legend(loc='lower right', frameon=True, edgecolor='black', bbox_to_anchor=(0.75, 0.))
# plt.title('(b)', y=-0.18, fontstyle='normal', fontweight='light')
# # plt.plot(x, h_fixed)
# # plt.plot(x, h_center)
# # plt.plot(x, h_independent)
# # plt.plot(x, h_rnn)
# # plt.plot(x, h_our)
plt.grid(alpha=0.9, linestyle='-.', which='both')
# plt.ylim(0,2400)
plt.text(7020, 107, 'sum', bbox=dict(boxstyle='square,pad=0.3', fc='linen', ec='k',lw=1,alpha=0.5))
plt.text(7000, 101, '3655')
plt.text(7000, 95, '3651')
plt.text(7000, 89, '3659')
plt.text(7000, 83, '3650')
plt.text(7000, 77, '3656')
plt.text(7000, 71, '3670')
#plt.show()
plt.savefig('throughtput.pdf')