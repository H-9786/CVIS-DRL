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

STATE_DIM = 5      # [车速，车辆加速度，车辆距离交叉口位置，信号灯状态]
ACTION_DIM = 1     # [加速度]
STATE_DIM_LIGHT = 4
STATE_DIM_LIGHT_SMALL = 3    # [车的距离,速度,加速度]
MAX_DIS = 300      # 与前车最大距离设定为道路长度, m
MAX_SPEED = 15     # 道路允许最大车速, m/s
Light_LOC = 13.6   # 路口停车线位置, m
RELATIVE_DIS = 30  # 两车独立控制的最大间隔
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

    # def reset(self):
    #     self.control.start([sumoBinary, "-c", "env.sumocfg"])
    #     self.time = 0
    #     self.light_time = [20, 3, 2, 20, 3, 2, 20, 3, 2, 20, 3, 2]
    #     self.lane_id = list(traci.lane.getIDList())
    #     node_id = []
    #     for i in range(len(self.lane_id)):
    #         if 'edge' not in self.lane_id[i]:
    #             node_id.append(self.lane_id[i])
    #     for i in range(len(node_id)):
    #         self.lane_id.remove(node_id[i])
    #     self.done = False

    def reset_light(self, sumoBinary):
        self.control.start([sumoBinary, "-c", "env.sumocfg"])
        return np.zeros(STATE_DIM_LIGHT)

    def get_car_name(self):
        return traci.vehicle.getIDList()

    def get_light_link(self, id):
        return traci.trafficlight.getControlledLinks(id)

    def get_sensor_data(self, sensor):
        return traci.inductionloop.getLastStepVehicleNumber(sensor)

    def get_car_position(self, car_name):
        return traci.vehicle.getPosition(car_name)

    def get_car_speed(self, car_name):
        return traci.vehicle.getSpeed(car_name)

    def get_car_fuelconsumption(self, car_name):
        return traci.vehicle.getFuelConsumption(car_name)

    def get_car_co2(self, car_name):
        return traci.vehicle.getCO2Emission(car_name)

    def set_cars_acc(self, car_name, car_acc):
        state = self.get_car_state(car_name)
        # print('前面：', car_name, state, car_acc)
        if state[0] > 30/300:
            # print("执行控制车速操作：", car_name)
            # print('控制：', car_name, state, car_acc)
            traci.vehicle.setSpeedMode(car_name, 1110)  # 更改车辆速度模型，使得能够控制其速度
            traci.vehicle.setLaneChangeMode(car_name, 0b000000000000)
            if car_acc >= 0:                            # 加速操作
                traci.vehicle.setAccel(car_name, car_acc)
                traci.vehicle.setSpeed(car_name, 15)
            else:                                       # 减速操作
                traci.vehicle.setDecel(car_name, np.abs(car_acc))
                traci.vehicle.setSpeed(car_name, 0)
            return car_name
        else:
            # car_acc = -state[0]
            # traci.vehicle.setSpeedMode(car_name, 1110)  # 更改车辆速度模型，使得能够控制其速度
            # traci.vehicle.setLaneChangeMode(car_name, 0b000000000000)
            # traci.vehicle.setDecel(car_name, np.abs(car_acc))
            # traci.vehicle.setSpeed(car_name, 0)
            traci.vehicle.setSpeedMode(car_name, 0xb3)
        # traci.vehicle.setSpeedMode(car_name, 00000)
            return None

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

    def get_car_state(self, car):
        # 每辆车的特定状态为（与前车距离，与前车速度差，与路口距离，当前车速，当前加速度，该相位绿灯时长）
        lane_id = traci.vehicle.getLaneID(car)                        # 该车所在车道
        position_car = traci.vehicle.getPosition(car)                 # 该车位置
        other_car = list(traci.lane.getLastStepVehicleIDs(lane_id))   # 该车所在车道所有车辆
        other_car.remove(car)                                         # 该车所在车道其余车辆

        # for i in other_car:
        #     print('当前车，其他车及其车速', car, traci.vehicle.getSpeed(car), i, traci.vehicle.getSpeed(i))

        current_speed = traci.vehicle.getSpeed(car)
        current_acc = traci.vehicle.getAcceleration(car)

        relative_dis = MAX_DIS         # 默认与前车相对距离为最大道路长度 300m
        relative_speed = MAX_SPEED     # 默认与前车相对速度为最大速度 15m/s
        cross_dis = MAX_DIS            # 默认与路口相对距离为最大道路长度 300m

        # 计算前车距离和前车相对速度
        if other_car and traci.vehicle.getRoadID(car) in ['edge_0', '-edge_2']: # 不同的道路有不同的距离比较方式
            relative_dis_mid = MAX_DIS     # 设相对距离的中间状态为最大道路长度
            for o_car in other_car:        # 遍历该道路上的其他车辆
                if position_car[0] < traci.vehicle.getPosition(o_car)[0]:       # 筛选出在目标车辆前面的车（x大的在该车前面）
                    # 寻找最小车距即为前车距离
                    relative_dis = min(abs(position_car[0] - traci.vehicle.getPosition(o_car)[0]), relative_dis_mid)
                    if abs(position_car[0] - traci.vehicle.getPosition(o_car)[0]) < relative_dis_mid:
                        relative_speed = traci.vehicle.getSpeed(o_car) - traci.vehicle.getSpeed(car) # 与前车相对车速
                relative_dis_mid = relative_dis
        if other_car and traci.vehicle.getRoadID(car) in ['-edge_1', 'edge_3']:
            relative_dis_mid = MAX_DIS
            for o_car in other_car:
                if position_car[1] < traci.vehicle.getPosition(o_car)[1]:
                    relative_dis = min(abs(position_car[1] - traci.vehicle.getPosition(o_car)[1]), relative_dis_mid)
                    if abs(position_car[1] - traci.vehicle.getPosition(o_car)[1]) < relative_dis_mid:
                        relative_speed = traci.vehicle.getSpeed(o_car) - traci.vehicle.getSpeed(car)
                relative_dis_mid = relative_dis
        if other_car and traci.vehicle.getRoadID(car) in ['-edge_0', 'edge_2']:
            relative_dis_mid = MAX_DIS
            for o_car in other_car:
                if position_car[0] > traci.vehicle.getPosition(o_car)[0]:
                    relative_dis = min(abs(position_car[0] - traci.vehicle.getPosition(o_car)[0]), relative_dis_mid)
                    if abs(position_car[0] - traci.vehicle.getPosition(o_car)[0]) < relative_dis_mid:
                        relative_speed = traci.vehicle.getSpeed(o_car) - traci.vehicle.getSpeed(car)
                relative_dis_mid = relative_dis
        if other_car and traci.vehicle.getRoadID(car) in ['edge_1', '-edge_3']:
            relative_dis_mid = MAX_DIS
            for o_car in other_car:
                if position_car[1] > traci.vehicle.getPosition(o_car)[1]:
                    relative_dis = min(abs(position_car[1] - traci.vehicle.getPosition(o_car)[1]), relative_dis_mid)
                    if abs(position_car[1] - traci.vehicle.getPosition(o_car)[1]) < relative_dis_mid:
                        relative_speed = traci.vehicle.getSpeed(o_car) - traci.vehicle.getSpeed(car)
                relative_dis_mid = relative_dis

        # 计算车辆距离路口距离
        if traci.vehicle.getRoadID(car) in ['edge_0', 'edge_2']:
            cross_dis = abs(traci.vehicle.getPosition(car)[0]) - Light_LOC
        elif traci.vehicle.getRoadID(car) in ['edge_1', 'edge_3']:
            cross_dis = abs(traci.vehicle.getPosition(car)[1]) - Light_LOC
        else:
            cross_dis = -1          # 驶离路口不受红绿灯控制，距离设置为-1

        # 获取当前车辆相位红绿灯状态
        light_time = self.light_time[::3]
        light_car = [0, 0, 0, 0]
        if traci.vehicle.getLaneID(car) in ['edge_1_1', 'edge_3_1']:
            light_car = [1, 0, 0, 0]
        if traci.vehicle.getLaneID(car) in ['edge_1_2', 'edge_3_2']:
            light_car = [1, 1, 0, 0]
        if traci.vehicle.getLaneID(car) in ['edge_0_1', 'edge_2_1']:
            light_car = [1, 1, 1, 0]
        if traci.vehicle.getLaneID(car) in ['edge_0_2', 'edge_2_2']:
            light_car = [1, 1, 1, 1]
        light_car_state = [a * b / 20 for a, b in zip(light_car, light_time)]

        if traci.vehicle.getLaneID(car) not in ['edge_1_1', 'edge_3_1', 'edge_1_2', 'edge_3_2',
                                                'edge_0_1', 'edge_2_1', 'edge_0_2', 'edge_2_2']:
            light_car_state = [-1, -1, -1, -1]

        # state = np.array([relative_dis/MAX_DIS, relative_speed/MAX_SPEED, cross_dis/MAX_DIS,
        #                   current_speed/MAX_SPEED, light_car_state])
        car_state = [relative_dis/300, relative_speed/15, cross_dis/300, current_speed/15, current_acc]#.extend(light_car_state)
        # state = [relative_dis, relative_speed, cross_dis, current_speed, current_acc]
        car_state.extend(light_car_state)
        # print(state)
        # state = np.array([relative_dis, relative_speed, cross_dis,
        #                   current_speed, light_car_state])

        return car_state

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

    def get_light_state_small(self, link):
        car_list_0 = traci.lane.getLastStepVehicleIDs(link[0])
        car_list_1 = traci.lane.getLastStepVehicleIDs(link[1])
        car_name = None
        Car_dis = 1
        for car in car_list_0:
            car_state = self.get_car_state(car)
            car_dis = car_state[2]    # 距离路口的距离
            if car_dis < Car_dis:
                Car_dis = car_dis
                car_name = car
        for car in car_list_1:
            car_state = self.get_car_state(car)
            car_dis = car_state[2]  # 距离路口的距离
            if car_dis < Car_dis:
                Car_dis = car_dis
                car_name = car
        if car_name == None:
            return None, np.zeros(STATE_DIM_LIGHT_SMALL)
        else:
            state_small = self.get_car_state(car_name)
            return car_name, [state_small[2], state_small[3], state_small[4]]
        #print(car_name, self.get_car_state(car_name)[2]*300)


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
        # lane_name = ['edge_1_1', 'edge_3_1', 'edge_1_2', 'edge_3_2', 'edge_0_1', 'edge_2_1', 'edge_0_2', 'edge_2_2']
        # waitingtime = []
        # speed = []
        # haltnum = []
        # for lane in lane_name:
        #     waitingtime.append(traci.lane.getWaitingTime(lane))
        #     speed.append(1 - traci.lane.getLastStepMeanSpeed(lane)/15)
        #     haltnum.append(traci.lane.getLastStepHaltingNumber(lane))
        # reward = -sum(waitingtime)*0.02-sum(speed)-sum(haltnum)/5
        # return reward

    def get_light_reward_small(self):
        pass

    def get_light_reward_new(self, state, state_next):
        halting_num_before = sum(state)
        halting_num_next = sum(state_next)
        # halting_num = -(halting_num_next - halting_num_before)
        if halting_num_next == 0:
            reward_halt = 5
        else:
            reward_halt = halting_num_before / halting_num_next

        # waiting_time_before = sum(state[8:])
        # waiting_time_next = sum(state_next[8:])
        # waiting_time = -(waiting_time_next - waiting_time_before)

        reward = reward_halt #+ waiting_time
        # reward = halting_num

        return reward


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
