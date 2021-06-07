import numpy as np

import matplotlib.pyplot as plt
from Environment import Environment
from RL_brain import DQN
import matplotlib.pyplot as plt

MAX_EPISODES = 1     # 训练回合数
STATE_DIM_LIGHT = 4    # 状态维度:四个相位的停车数目
ACTION_DIM_LIGHT = 4   # [周期,四个相位的绿信比]
MAX_ACTION = 1         # 输出动作边界
MEMORY_CAPACITY = 5000 # 记忆库大小
MAX_DIS = 300          # 与前车最大距离设定为道路长度, m
MAX_SPEED = 15         # 道路允许最大车速, m/s
MIN_CYCLE = 40         # 除去黄灯和红灯的最小周期长度
TIME = 9000            # 一个回合仿真时长

env = Environment()    # 环境

dqn = DQN()   # 智能体
dqn.load("../model/dqn_light40")
var_l = 1.0            # 方差,用于指示智能体是否进行学习

# 指示相位序号和其所控制的道路编号的对应关系
phase_to_link = {0: ['edge_1_1', 'edge_3_1'],
                 3: ['edge_1_2', 'edge_3_2'],
                 6: ['edge_0_1', 'edge_2_1'],
                 9: ['edge_0_2', 'edge_2_2']}

# 传感器编号:用以统计每一时刻离开路网的车辆数目
sensors = ['e_0_0', 'e_0_1', 'e_0_2', 'e_1_0', 'e_1_1', 'e_1_2',
           'e_2_0', 'e_2_1', 'e_2_2', 'e_3_0', 'e_3_1', 'e_3_2']

# 存储文件函数:将列表数据存储为txt文档
def text_save(filename, data):
    file = open(filename, 'w+')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')
        s = s.replace("'", '').replace(',', '') + '\n'
        file.write(s)
    file.close()

# 循环移位函数:用来进行状态\动作的循环移位,扩充记忆库
def list_move_left(A, a):
    A = list(A)
    for i in range(a):
        A.insert(len(A), A[0])
        A.remove(A[0])
    return A

from sumolib import checkBinary

reward_list = []     # 用以存储每一回合奖励值变化情况
waiting_list = []    # 用以存储每一回合等待时间变化情况
jam_list = []        # 用以存储每一回合停车数目变化情况

car_speed_analysis = [['car_'+str(i+1)] for i in range(3600)]
car_co2_analysis = [['car_'+str(i+1)] for i in range(3600)]
car_fuel_analysis = [['car_'+str(i+1)] for i in range(3600)]
car_acc_analysis = [['car_'+str(i+1)] for i in range(3600)]

for episode in range(MAX_EPISODES):      # 进行每一回合的训练学习
    # change.rewrite(time_period=[(0, 3000), (3000, 6000), (6000, 9000)], car_num=[1000, 1500, 1200])

    # 是否需要显示图形界面,若需要显示,将'sumo'改为'sumo-gui'即可
    if episode >= 10 and episode % 10 == 0:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo')

    state_light = env.reset_light(sumoBinary)     # 初始化环境,并返回最初的状态值:全零向量
    ep_reward = 0                                 # 用以统计该回合总奖励
    throughput = []                               # 用以统计该回合吞吐量变化情况
    ep_jam, ep_waiting = [], []                   # 用以统计该回合等待时间和停车数目变化情况
    time_episode = TIME                           # 该回合总持续时间
    while time_episode >= 0:                      # 当该回合持续时间并未结束,执行:
        action = dqn.choose_action(state_light)
        #print(action)
        light_duration = [0,0,0,0,0,0,0,0,0,0,0,0]
        light_duration[action*3] = 15
        light_duration[action*3+1] = 2
        light_duration[action*3+2] = 3

        # 指定回合对动作输出和配时方案进行打印
        # if episode >= 10 and episode % 10 == 0:
        #     print(action)
        #     print('light_duration:', light_duration, sum(light_duration))

        time = 0   # 针对该周期进行计时的指示器
        while time < sum(light_duration):   # 当该周期并未结束时
            cars = env.get_car_name()   # 获取路网中的所有车辆
            for car in cars:            # 遍历车辆
                if car == 'car_1':
                    print(env.get_car_speed(car))
                car_speed = env.get_car_speed(car) # 获取该车速度信息
                car_co2 = env.get_car_co2(car)
                car_fuel = env.get_car_fuelconsumption(car)

                for car_i in range(len(car_speed_analysis)):  # 将该车速度信息存储列表对应位置
                    if car == car_speed_analysis[car_i][0]:
                        car_speed_analysis[car_i].append(car_speed)
                        car_co2_analysis[car_i].append(car_co2)
                        car_fuel_analysis[car_i].append(car_fuel)
                        if len(car_speed_analysis[car_i]) >= 3:
                            car_acc_analysis[car_i].append(abs(car_speed_analysis[car_i][-1] - car_speed_analysis[car_i][-2]))

            throughput_time = 0
            for sensor in sensors:    # 统计该时刻(s)检测器检测通过路网的车辆数
                throughput_time += env.get_sensor_data(sensor)
            throughput.append(throughput_time)

            for i in range(len(light_duration) + 1):  # 找到当前时刻所对应的相位编号
                if time < sum(light_duration[:i]):
                    index = i-1
                    break

            env.change_light(index, light_duration[index])  # 对红绿灯进行控制
            env.step()                                      # 更新一步环境
            wait, jam = env.get_totol_reward()              # 得到当前时刻的等待时间和停车数目信息
            ep_waiting.append(wait)                         # 将当前时刻等待时间存入该回合等待时间列表
            ep_jam.append(jam)                              # 将当前时刻停车数目存入该回合停车数目列表
            time += 1                                       # 当前周期时间步长+1
            time_episode -= 1                               # 该回合总持续时间-1

        state_light_next = env.get_light_state()            # 获取下一时刻状态值
        reward_light = env.get_light_reward()               # 获取奖励信息
        # 记忆库存储,并通过翻转动作,状态信息进行数据扩充,增加数据样本量
        dqn.store_transition(state_light, action, reward_light, state_light_next)

        state_light = state_light_next   # 更新当前状态
        ep_reward += reward_light        # 将当前奖励值累加到回合奖励值

        if dqn.memory_counter > MEMORY_CAPACITY:   # 当记忆库存储满
            var_l *= 0.9995
            var_l = max(0.01, var_l)
            dqn.learn()                     # 进行智能体学习

    print('Episode:', episode, 'reward:', ep_reward, 'epsilon:', dqn.epsilon,
          'waiting:', sum(ep_waiting)/9000, 'jam:', sum(ep_jam)/9000)     # 打印回合信息

    car_average_speed = 0
    car_average_time = 0
    car_average_co2 = 0
    car_average_fuel = 0
    car_average_acc = 0
    avg_speed = []
    avg_time = []
    avg_co2 = []
    avg_fuel = []
    avg_acc = []
    for i in range(len(car_speed_analysis)):
        car_average_time += (len(car_speed_analysis[i]) - 1)
        car_average_speed += sum(car_speed_analysis[i][1:]) / (len(car_speed_analysis[i]) - 1)
        car_average_co2 += sum(car_co2_analysis[i][1:])
        car_average_fuel += sum(car_fuel_analysis[i][1:])
    for i in range(len(car_acc_analysis)):
        car_average_acc += sum(car_acc_analysis[i][1:]) / (len(car_acc_analysis[i]) - 1)

    car_average_time /= len(car_speed_analysis)
    car_average_speed /= len(car_speed_analysis)
    car_average_co2 /= len(car_co2_analysis)
    car_average_fuel /= len(car_fuel_analysis)
    car_average_acc /= len(car_acc_analysis)
    avg_speed.append(car_average_speed)
    avg_time.append(car_average_time)
    avg_co2.append(car_average_co2)
    avg_fuel.append(car_average_fuel)
    avg_acc.append(car_average_acc)
    print(car_average_time, car_average_speed, car_average_co2, car_average_fuel, car_average_acc)
    # print(car_analysis[0])
    reward_list.append(ep_reward)
    waiting_list.append(sum(ep_waiting) / 9000)
    jam_list.append(sum(ep_jam) / 9000)
    text_save('jam_9000.txt', ep_jam)
    text_save('wait_9000.txt', ep_waiting)
    text_save('throughtput_test.txt', throughput)
    text_save('avg_speed.txt', avg_speed)
    text_save('avg_time.txt', avg_time)
    text_save('avg_co2.txt', avg_co2)
    text_save('avg_fuel.txt', avg_fuel)
    text_save('avg_acc.txt', avg_acc)
    env.end()

text_save('reward_light.txt', reward_list)
text_save('jam.txt', jam_list)
text_save('wait.txt', waiting_list)

