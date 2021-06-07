import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import sys
import optparse
import numpy as np
import re
import copy
import random
from random import choice
import matplotlib.pyplot as plt
import math
from Environment import Environment
from RL_brain import TD3
#from Change_Xml import ChangeXml
import matplotlib.pyplot as plt

MAX_EPISODES = 1

STATE_DIM_LIGHT = 4    # 8条车道的排队车辆数和等待时间信息
ACTION_DIM_LIGHT = 5   # [周期, 各相位绿信比]
ACTION_DIM_LIGHT_SMALL = 1   # [决定是否加减时长]
STATE_DIM_LIGHT_SMALL = 3    # [车的距离,速度,加速度]
MAX_ACTION = 1
MEMORY_CAPACITY = 8000
MEMORY_CAPACITY_S = 8000
MAX_DIS = 300    # 与前车最大距离设定为道路长度, m
MAX_SPEED = 15   # 道路允许最大车速, m/s
MIN_CYCLE = 40   # 最大周期长度
TIME = 9000
GAMMA = 0.9
GAMMA_S = 0.1
env = Environment()
#change = ChangeXml()
expl_noise = 0.0
td3_light = TD3(STATE_DIM_LIGHT, ACTION_DIM_LIGHT, 1., MEMORY_CAPACITY, GAMMA, lr_a=0.0001, lr_c=0.0001)
td3_light_small = TD3(STATE_DIM_LIGHT_SMALL, ACTION_DIM_LIGHT_SMALL, 1., MEMORY_CAPACITY_S, GAMMA_S, lr_a=0.0001, lr_c=0.001)
td3_light.load('../model/td3_light160')
td3_light_small.load('../model/td3_light_small160')
print(td3_light.critic)

var_l = 0.1
var_l_s = 0.1
ep_reward_save = []

phase_to_link = {0: ['edge_1_1', 'edge_3_1'],
                 3: ['edge_1_2', 'edge_3_2'],
                 6: ['edge_0_1', 'edge_2_1'],
                 9: ['edge_0_2', 'edge_2_2']}

sensors = ['e_0_0', 'e_0_1', 'e_0_2', 'e_1_0', 'e_1_1', 'e_1_2',
           'e_2_0', 'e_2_1', 'e_2_2', 'e_3_0', 'e_3_1', 'e_3_2']


def text_save(filename, data):
    file = open(filename, 'w+')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')
        s = s.replace("'", '').replace(',', '') + '\n'
        file.write(s)
    file.close()

def list_move_left(A, a):
    # a = list(a)
    A = list(A)
    for i in range(a):
        A.insert(len(A), A[0])
        A.remove(A[0])
    return A

from sumolib import checkBinary

reward_list = []
reward_s_list = []
waiting_list = []
jam_list = []

car_speed_analysis = [['car_'+str(i+1)] for i in range(3600)]
car_co2_analysis = [['car_'+str(i+1)] for i in range(3600)]
car_fuel_analysis = [['car_'+str(i+1)] for i in range(3600)]
car_acc_analysis = [['car_'+str(i+1)] for i in range(3600)]

for episode in range(MAX_EPISODES):
    print('actor_lr:', td3_light.actor_optimizer.param_groups[0]['lr'],
          'critic_lr:', td3_light.critic_optimizer.param_groups[0]['lr'],
          'actor_s_lr:', td3_light_small.actor_optimizer.param_groups[0]['lr'],
          'critic_s_lr:', td3_light_small.critic_optimizer.param_groups[0]['lr'])

    sumoBinary = checkBinary('sumo')

    state_light = env.reset_light(sumoBinary)
    ep_reward = 0
    ep_reward_s = 0
    throughput = []
    time_episode = TIME
    ep_jam, ep_waiting = [], []
    while time_episode >= 0:

        action_l = (td3_light.choose_action(np.array(state_light)) +
                    np.random.normal(0, MAX_ACTION * expl_noise, size=ACTION_DIM_LIGHT)).clip(-MAX_ACTION, MAX_ACTION)
        action_l_v = (action_l + 1) / 2

        cycle = round(action_l_v[0] * 60 + MIN_CYCLE)  # 控制周期在40-100区间内

        green_ratio = []
        green_ratio = [round(action_l_v[i+1] / (sum(action_l_v[1:])+0.0001) * cycle) for i in range(len(action_l_v) - 1)]

        green_ratio[0] += cycle - sum(green_ratio)


        light_duration = [max(5, green_ratio[0]), 2, 3, max(5, green_ratio[1]), 2, 3,
                          max(5, green_ratio[2]), 2, 3, max(5, green_ratio[3]), 2, 3]

        flag_phase = [False] * 12
        time = 0
        while time < sum(light_duration):
            cars = env.get_car_name()  # 获取路网中的所有车辆
            for car in cars:  # 遍历车辆
                if car == 'car_1':
                    print(env.get_car_speed(car))
                car_speed = env.get_car_speed(car)  # 获取该车速度信息
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
            for sensor in sensors:
                throughput_time += env.get_sensor_data(sensor)
            throughput.append(throughput_time)

            for i in range(len(light_duration) + 1):
                if time < sum(light_duration[:i]):
                    index = i-1
                    break

            for i in range(len(light_duration)):
                if (time == sum(light_duration[:i+1])-5) and (i==0 or i==3 or i==6 or i==9) and flag_phase[i] == False:
                    link = phase_to_link[i]  # 找到当前相位所对应的道路编号
                    car, state_light_small = env.get_light_state_small(link)
                    # print(car, state_light_small)
                    if car == None:
                        action_l_s = -1
                    else:
                        action_l_s = (td3_light_small.choose_action(np.array(state_light_small)) +
                                      np.random.normal(0, MAX_ACTION * expl_noise, size=ACTION_DIM_LIGHT_SMALL)).clip(-MAX_ACTION, MAX_ACTION)
                    # action_l_s = 0
                    light_duration[i] += round(float(action_l_s) * 3)

                    flag_phase[i] = True

                if time == sum(light_duration[:i+1]) and (i==0 or i==3 or i==6 or i==9) and car != None:
                    State_light_small_next = env.get_car_state(car)
                    state_light_small_next = [State_light_small_next[2], State_light_small_next[3], State_light_small_next[4]]
                    # print(car, state_light_small_next)
                    ideal_time = math.ceil(state_light_small[0] * MAX_DIS / (state_light_small[1] * MAX_SPEED + 1)) + 3
                    do_time = round(float(action_l_s) * 3)

                    if ideal_time > 8:
                        reward_light_small = -do_time # -3,3
                    elif 5 < ideal_time <= 8:
                        reward_light_small = -abs(ideal_time - (do_time + 5))
                    else:
                        reward_light_small = -abs(ideal_time - (5 + do_time))

                    if state_light_small_next[0] < 0:
                        reward_light_small += 10
                    if state_light_small_next[0] >= 0 and do_time == 3:
                        reward_light_small -= 5

                    #print(state_light_small, action_l_s, reward_light_small, state_light_small_next)
                    td3_light_small.store_transition(state_light_small, action_l_s, reward_light_small, state_light_small_next)
                    ep_reward_s += reward_light_small

                    if td3_light_small.pointer > MEMORY_CAPACITY_S:
                        var_l_s *= 0.999
                        var_l_s = max(0.01, var_l_s)
                        td3_light_small.learn()

            env.change_light(index, light_duration[index])
            env.step()
            wait, jam = env.get_totol_reward()
            ep_waiting.append(wait)
            ep_jam.append(jam)
            time += 1
            time_episode -= 1

        state_light_next = env.get_light_state()
        reward_light = env.get_light_reward()

        state_light = state_light_next
        ep_reward += reward_light

    print('Episode:', episode, 'reward_l:', ep_reward, 'var_l: %.2f' % var_l, 'reward_s:', ep_reward_s, 'var_s: %.2f' % var_l_s,
          'waiting:', sum(ep_waiting)/9000, 'jam:', sum(ep_jam)/9000)

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