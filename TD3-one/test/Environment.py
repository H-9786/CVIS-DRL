import os
import sys
import optparse
import time
import numpy as np
np.set_printoptions(threshold=np.inf)

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME")

import traci
from sumolib import checkBinary

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true", default=False, help="run the commandline version of sumo")

    options, args = optParser.parse_args()
    return options


STATE_DIM_LIGHT = 4
MAX_DIS = 300      # 与前车最大距离设定为道路长度, m
MAX_SPEED = 15     # 道路允许最大车速, m/s
Light_LOC = 13.6   # 路口停车线位置, m
ETA = 0.15         # 奖励有关常量
ROU = 2            # 奖励有关常量
C = 20             # 奖励有关常量



class Environment(object):
    def __init__(self):
        self.control = traci
        self.time = 0
        self.car_name = None
        self.light_time = [20, 3, 2, 20, 3, 2, 20, 3, 2, 20, 3, 2]
        self.lane_id = None
        self.done = False

    def reset_light(self, sumoBinary):
        self.control.start([sumoBinary, "-c", "env.sumocfg"])
        return np.zeros(STATE_DIM_LIGHT)

    def get_car_name(self):
        return traci.vehicle.getIDList()

    def get_light_link(self, id):
        return traci.trafficlight.getControlledLinks(id)

    def get_car_position(self, car_name):
        return traci.vehicle.getPosition(car_name)

    def get_car_speed(self, car_name):
        return traci.vehicle.getSpeed(car_name)

    def get_car_co2(self, car_name):
        return traci.vehicle.getCO2Emission(car_name)

    def get_car_fuelconsumption(self, car_name):
        return traci.vehicle.getFuelConsumption(car_name)

    def get_sensor_data(self, sensor):
        return traci.inductionloop.getLastStepVehicleNumber(sensor)

    def get_env_state(self):
        # 返回整个环境的全局状态信息（车辆位置信息以及车速信息，构成一个矩阵，通过 CNN 进行处理）
        env_state = np.zeros((24, 60))
        # car_id = traci.vehicle.getIDList()
        lane_id = self.lane_id     # 获取道路名称
        count = 0                  # 遍历道路时的计数
        for lane in lane_id:
            car_id = traci.lane.getLastStepVehicleIDs(lane)      # 获取该条道路上的车辆名称
            for car in car_id:        # 遍历该条道路上的每辆车，根据相应位置将车速信息填入路网状态矩阵
                pos = traci.vehicle.getPosition(car)
                if lane[0] == '-' and (lane[6] == '0' or lane[6] == '2'):
                    pos = int((abs(pos[0]) - Light_LOC) // 5)
                    car_speed = traci.vehicle.getSpeed(car)
                    env_state[count][pos] = -car_speed
                if lane[0] != '-' and (lane[5] == '0' or lane[5] == '2'):
                    pos = int((abs(pos[0]) - Light_LOC) // 5)
                    car_speed = traci.vehicle.getSpeed(car)
                    env_state[count][pos] = car_speed
                if lane[0] == '-' and (lane[6] == '1' or lane[6] == '3'):
                    pos = int((abs(pos[1]) - Light_LOC) // 5)
                    car_speed = traci.vehicle.getSpeed(car)
                    env_state[count][pos] = -car_speed
                if lane[0] != '-' and (lane[5] == '1' or lane[5] == '3'):
                    pos = int((abs(pos[1]) - Light_LOC) // 5)
                    car_speed = traci.vehicle.getSpeed(car)
                    env_state[count][pos] = car_speed
            count += 1
        # env_state = None
        return env_state

    def get_light_state(self):
        state_light = []
        lane_name = ['edge_1_1', 'edge_3_1', 'edge_1_2', 'edge_3_2', 'edge_0_1', 'edge_2_1', 'edge_0_2', 'edge_2_2']
        for lane in lane_name:
            state_light.append(traci.lane.getLastStepHaltingNumber(lane)/10)
        # for lane in lane_name:
        #     state_light.append(traci.lane.getWaitingTime(lane)/50)
        state = [state_light[0]+state_light[1], state_light[2]+state_light[3],
                 state_light[4]+state_light[5], state_light[6]+state_light[7]]
        return state


    def get_light_reward(self):
        ID_list = traci.vehicle.getIDList()
        # print(ID_list, type(ID_list))
        reward_light = 0
        for id in range(len(ID_list)):
            id_car = ID_list[id]
            waiting_time = traci.vehicle.getWaitingTime(id_car)
            reward_light += ETA * (1 - (waiting_time / C) ** ROU)
        if len(ID_list) == 0:
            reward = 0
        else:
            reward_light /= len(ID_list)
        return reward_light

    def step_env(self):
        traci.simulationStep()
        if sum(self.light_time) == 0:
            self.light_time = [20, 3, 2, 20, 3, 2, 20, 3, 2, 20, 3, 2]
        for t in range(len(self.light_time)):
            if self.light_time[t] != 0:
                self.light_time[t] -= 1
                break

        self.time += 1

    def step(self):
        traci.simulationStep()

    def end(self):    # 结束sumo进程
        self.control.close()

    def change_light(self, index, duration):
        traci.trafficlight.setPhase('gneJ1', index)
        traci.trafficlight.setPhaseDuration('gneJ1', duration)

    def get_totol_reward(self):
        waiting_time, jam_length = 0, 0
        edges = ['edge_0_1', 'edge_0_2', 'edge_1_1', 'edge_1_2',
                 'edge_2_1', 'edge_2_2', 'edge_3_1', 'edge_3_2']
        for edge in edges:
            waiting_time += traci.lane.getWaitingTime(edge)
            jam_length += traci.lane.getLastStepHaltingNumber(edge)
        return waiting_time/8, jam_length/8


if __name__ == "__main__":
    env = Environment()
    env.reset()
    state_list = []
    action_list = []

    def get_reward(state_list):
        pre_dis = state_list[0][-2][0]
        pre_speed = state_list[0][-2][1]
        if pre_speed == 0:
            TTC = 0
        else:
            TTC = -pre_dis / pre_speed
        reward_ttc = [np.log10(TTC / 4) if 0 < TTC <= 4 else 0]
        reward_speed = state_list[0][-2][3]

        reward_comf = [abs(1 / (state_list[0][-3][4] - state_list[0][-2][4]))
                       if (state_list[0][-3][4] - state_list[0][-2][4]) != 0 else 1]

        reward = reward_comf[0] + reward_ttc[0] + reward_speed

        if state_list[0][-2][3] == 0:
            reward = -1
        return reward

    for i in range(100):
        if not env.get_car_name():  # 当环境中没有车辆时，直接进行全局路网更新
            env.step_env()
        else:                       # 当环境中存在车辆时，对每个车进行控制
            for car in env.get_car_name():
                acc = np.random.randn(1)
                car_exit = [name[0] for name in action_list]  # 查找现阶段动作列表中存在的车辆
                if car not in car_exit:                       # 如果车辆不在现阶段存在列表中，则加入新的车辆到车辆列表
                    action_list.append([car])
                car_exit = [name[0] for name in state_list]   # 查找现阶段存在车辆
                index = car_exit.index(car)                   # 找到该车辆对应在车辆列表中的位置
                action_list[index].append(acc)                # 将动作信息存入动作列表
                # print(car, action_list[index][1:])
                env.set_cars_acc(car, acc)
            env.step_env()

        for car in env.get_car_name():
            car_exit = [name[0] for name in state_list]    # 查找现阶段状态列表中存在的车辆
            if car not in car_exit:                        # 如果车辆不在现阶段存在列表中，则加入新的车辆到车辆列表
                state_list.append([car])

            car_exit = [name[0] for name in state_list]    # 查找现阶段存在车辆
            index = car_exit.index(car)                    # 找到该车辆对应在车辆列表中的位置
            state_car = env.get_car_state(car)                 # 获取该车辆状态信息
            state_light = env.get_light_state()
            state_env = env.get_env_state()
            print(car, state_car, state_light)
            # print(state_env)
            # print('动作列表：', action_list)
            # if action_list:
            #     print('动作列表：', action_list[index])

            state_list[index].append(state_car)            # 将该车辆状态信息存如车辆状态列表


            # if car == 'car_1':
            #     print(state_car, reward_car)

        if state_list and len(state_list[0]) > 3 and len(action_list[0]) >= 3:
            reward = get_reward(state_list)
            action_list[0][-2] = state_list[0][-2][4]
            transition = [state_list[0][0], state_list[0][-3], action_list[0][-2], reward, state_list[0][-2]]
            print('transition:', transition)

            # print(car, state_car)
            # print(state_list)
    # print(state_list, action_list, reward_list)
        #env.step_env()
    # env.step_env()
