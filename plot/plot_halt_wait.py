import numpy as np
import matplotlib.pyplot as plt

def text_read(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    x = []

    for line in lines:
        x.append(float(line.split('\n')[0]))

    file.close()
    return x

def add(y):
    x = []
    for i in range(30):
        x.append(sum(y[i*300:i*300+300])/300)
        #print(x)
    return x

figsize = 14, 5
plt.subplots(figsize=figsize)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.15)
plt.rcParams['figure.figsize'] = (11, 4)

plt.subplot(1,2,1)
fix_agent = text_read('jam_fix.txt')
fix_agent = add(fix_agent)

long_agent = text_read('jam_long.txt')
long_agent = add(long_agent)

dqn_agent = text_read('jam_dqn.txt')
dqn_agent = add(dqn_agent)

one_agent = text_read('jam_td3_one.txt')
one_agent = add(one_agent)

two_agent = text_read('jam_td3_two.txt')
two_agent = add(two_agent)

three_agent = text_read('jam_td3_three.txt')
three_agent = add(three_agent)

print(np.average(fix_agent), np.average(long_agent), np.average(dqn_agent),
      np.average(one_agent), np.average(two_agent), np.average(three_agent))

# print(len(h_fixed))
x = np.linspace(0, 9000, 30)

plt.plot(x, fix_agent, label='Fixed-time', marker='s', linestyle='dashed', color='orange')
plt.plot(x, long_agent, label='Longest-queue-first', color='indianred',marker='o', linestyle='dashed')
plt.plot(x, dqn_agent, label='DQN', marker='H', linestyle='dashed', color='plum')
plt.plot(x, one_agent, label='TD3-one-agent', marker='^', linestyle='dashed', color='turquoise')
plt.plot(x, two_agent, label='TD3-two-agent', marker='d', linestyle='dashed', color='dodgerblue')
plt.plot(x, three_agent, label='TD3-three-agent', marker='p', linestyle='dashed', color='blueviolet')
# plt.plot(x, w_spatial, label='Spatial-DDPG', marker='h', linestyle='dashed', color='mediumslateblue')
# plt.plot(x, w_our, label='MARL-DSTAN', marker='o',color='steelblue')

plt.ylabel('Average number of halting vehicles per lane / veh', fontsize=10)
plt.xlabel('Time / s', fontsize=12)
#plt.legend(loc='lower right', frameon=True, edgecolor='black', bbox_to_anchor=(0.75, 0.))
#plt.legend(loc='upper left', frameon=True, edgecolor='black')
#legends = {'random', 'SAA', 'A2C', 'A3C', 'TA3C'}
# plt.tight_layout()
# plt.legend(loc='upper left', bbox_to_anchor=(-0.3,1.15), frameon=False)
#plt.legend(bbox_to_anchor=(0.35, 1.115), frameon=True, edgecolor='black', ncol=5)
# plt.title('(b)', y=-0.18, fontstyle='normal', fontweight='light')
# # plt.plot(x, h_fixed)
# # plt.plot(x, h_center)
# # plt.plot(x, h_independent)
# # plt.plot(x, h_rnn)
# # plt.plot(x, h_our)
plt.grid(alpha=0.9, linestyle='-.', which='both')
# plt.ylim(0,2400)
# plt.text(7000, 107, 'sum', bbox=dict(boxstyle='square,pad=0.3', fc='linen', ec='k',lw=1,alpha=0.5))
# plt.text(7000, 101, '3655')
# plt.text(7000, 95, '3651')
# plt.text(7000, 89, '3659')
# plt.text(7000, 83, '3650')
# plt.text(7000, 77, '3656')
# plt.text(7000, 71, '3670')

plt.subplot(1,2,2)
fix_agent = text_read('wait_fix.txt')
fix_agent = add(fix_agent)

long_agent = text_read('wait_long.txt')
long_agent = add(long_agent)

dqn_agent = text_read('wait_dqn.txt')
dqn_agent = add(dqn_agent)

one_agent = text_read('wait_td3_one.txt')
one_agent = add(one_agent)

two_agent = text_read('wait_td3_two.txt')
two_agent = add(two_agent)

three_agent = text_read('wait_td3_three.txt')
three_agent = add(three_agent)

print(np.average(fix_agent), np.average(long_agent), np.average(dqn_agent),
      np.average(one_agent), np.average(two_agent), np.average(three_agent))

# print(len(h_fixed))
x = np.linspace(0, 9000, 30)

plt.plot(x, fix_agent, label='Fixed-time', marker='s', linestyle='dashed', color='orange')
plt.plot(x, long_agent, label='Longest-queue-first', color='indianred',marker='o', linestyle='dashed')
plt.plot(x, dqn_agent, label='DQN', marker='H', linestyle='dashed', color='plum')
plt.plot(x, one_agent, label='TD3-one-agent', marker='^', linestyle='dashed', color='turquoise')
plt.plot(x, two_agent, label='TD3-two-agent', marker='d', linestyle='dashed', color='dodgerblue')
plt.plot(x, three_agent, label='TD3-three-agent', marker='p', linestyle='dashed', color='blueviolet')
# plt.plot(x, w_spatial, label='Spatial-DDPG', marker='h', linestyle='dashed', color='mediumslateblue')
# plt.plot(x, w_our, label='MARL-DSTAN', marker='o',color='steelblue')

plt.ylabel('Average waiting time of vehicles per lane / s', fontsize=10)
plt.xlabel('Time / s', fontsize=12)
#plt.legend(loc='lower right', frameon=True, edgecolor='black', bbox_to_anchor=(0.75, 0.))
#plt.legend(loc='upper left', frameon=True, edgecolor='black')
#legends = {'random', 'SAA', 'A2C', 'A3C', 'TA3C'}
# plt.tight_layout()
# plt.legend(loc='upper left', bbox_to_anchor=(-0.3,1.15), frameon=False)
plt.legend(bbox_to_anchor=(0.85, 1.115), frameon=True, edgecolor='black', ncol=6)
# plt.title('(b)', y=-0.18, fontstyle='normal', fontweight='light')
# # plt.plot(x, h_fixed)
# # plt.plot(x, h_center)
# # plt.plot(x, h_independent)
# # plt.plot(x, h_rnn)
# # plt.plot(x, h_our)
plt.grid(alpha=0.9, linestyle='-.', which='both')
# plt.ylim(0,2400)
# plt.text(7000, 107, 'sum', bbox=dict(boxstyle='square,pad=0.3', fc='linen', ec='k',lw=1,alpha=0.5))
# plt.text(7000, 101, '3655')
# plt.text(7000, 95, '3651')
# plt.text(7000, 89, '3659')
# plt.text(7000, 83, '3650')
# plt.text(7000, 77, '3656')
# plt.text(7000, 71, '3670')
# plt.show()
plt.savefig('halt_wait.pdf')