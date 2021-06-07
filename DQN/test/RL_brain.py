"""
TD3 算法
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

# 设置超参数
LR = 0.0001                 # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
MEMORY_CAPACITY = 8000
BATCH_SIZE = 64
N_STATES = 4
N_ACTIONS = 4
TARGET_REPLACE_ITER = 100   # target update frequency
ENV_A_SHAPE = 0

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 400)
        #self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, N_ACTIONS)
        # self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions_value = self.fc3(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.epsilon = EPSILON
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < self.epsilon:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon *= 1.001
        self.epsilon = min(0.9, self.epsilon)

    def save(self, filename):
        torch.save(self.eval_net.state_dict(), filename+"_eval")
        torch.save(self.optimizer.state_dict(), filename+'_eval_optimizer')

    def load(self, filename):
        self.eval_net.load_state_dict(torch.load(filename+"_eval"))
        self.optimizer.load_state_dict(torch.load(filename+"_eval_optimizer"))
        self.target_net = copy.deepcopy(self.eval_net)
















































