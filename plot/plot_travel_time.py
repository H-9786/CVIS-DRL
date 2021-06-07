import matplotlib.pyplot as plt

X = ['Fixed-time', 'Longest-queue-first', 'DQN', 'TD3-one-agent', 'TD3-two-agent', 'TD3-three-agent']
Y = [69.94, 68.83, 65.08, 65.88, 63.73, 59.38]
fig = plt.figure()
plt.bar(X, Y, 0.4, color="steelblue", hatch='', edgecolor='black')
plt.xlabel("")
plt.ylabel("Average total travel time per veh / s")

for a, b in zip(X, Y):
 plt.text(a, b, '%.2f' % b, ha='center', va='bottom')
plt.xticks(rotation = 9.5)

plt.ylim(0, 75)

plt.grid(axis="y", linestyle='-.')
#plt.show()
plt.savefig('travel_time.pdf')

