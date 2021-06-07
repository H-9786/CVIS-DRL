import numpy as np
import random
from random import choice

# 根据入口和出口决定驶入车道
Dict = {'0,1': 2, '0,2': 1, '0,3': 0,
        '1,0': 0, '1,2': 2, '1,3': 1,
        '2,0': 1, '2,1': 0, '2,3': 2,
        '3,0': 2, '3,1': 1, '3,2': 0}

class ChangeXml(object):
    def __init__(self):
        pass

    def change(self, keyword, changeword):
        with open('rou.rou.xml', 'r', encoding='utf-8') as f:
            lines=[]
            for line in f.readlines():
                if line != '\n':
                    lines.append(line)
            f.close()

        with open('rou.rou.xml', 'w', encoding='utf-8') as f:
            for line in lines:
                if keyword in line:
                    f.write(changeword)
                else:
                    f.write('%s' % line)

    def rewrite(self, time_period, car_num):
        with open('rou.rou.xml', 'w', encoding='utf-8') as f:
            count = 0
            f.write('<routes>\n')
            # 写入车辆类型：最大加速度，最大减速度，车长度，最大速度
            f.write('  <vType accel="3" decel="8" id="CarA" length="5" maxSpeed="15" reroute="false" sigma="0.5" />\n\n')
            for time in range(len(time_period)):
                depart_time = np.random.randint(time_period[time][0], time_period[time][1], car_num[time])
                depart_time.sort()
                # print(depart_time)
                for car in range(car_num[time]):
                    count += 1
                    o, d = random.sample(range(0, 4), 2)   # 随机选择驶入口和驶出口
                    o_lane = Dict[str(o) + ',' + str(d)]   # 由于不考虑变道模型，故需要限制驶入车道
                    f.write('  <vehicle id="car_'+str(count)+'" color="1,1,0" depart="' + str(depart_time[car]) +
                            '" departLane="' + str(o_lane) + '" arrivalLane="' + str(np.random.randint(3)) +
                            '" departSpeed="max" type="CarA">\n    '
                            '<route edges="edge_' + str(o) + ' -edge_' + str(d) + '"/>\n  '
                            '</vehicle>\n')
                    f.write('\n')
            f.write('</routes>\n')


if __name__ == '__main__':
    change = ChangeXml()
    #change.rewrite([(0, 10), (10, 20), (20, 30)], [2, 5, 3])
    change.rewrite(time_period=[(0, 2000), (2000, 4000), (4000, 6000)], car_num=[500, 1000, 700])