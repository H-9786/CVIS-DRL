"""
TD3 算法
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

# 设置超参数
TAU = 0.005
BATCH_SIZE = 64

class Actor(nn.Module):           # 定义 actor 网络结构
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x

class Critic(nn.Module):          # 定义 critic 网络结构
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture   计算 Q1
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture   计算 Q2
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, memory_capacity, gamma, lr_a, lr_c):
        self.s_dim = state_dim       # 状态维度
        self.a_dim = action_dim      # 动作维度
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.memory_capacity = memory_capacity
        self.gamma = gamma
        self.memory = np.zeros((self.memory_capacity, state_dim * 2 + action_dim + 1), dtype=np.float32)  # 记忆库
        self.pointer = 0  # 记忆库存储数据指针

        # 创建对应的四个网络
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())    # 存储网络名字和对应参数
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)  # actor 优化器
        self.scheduler_actor = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=200, gamma=0.5)#500

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)
        self.scheduler_critic = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=400, gamma=0.5)#300

        self.max_action = max_action

        self.policy_freq = 2
        self.total_it = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.memory_capacity
        self.memory[index, :] = transition
        self.pointer += 1

    def choose_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()

    def learn(self):
        self.total_it += 1
        # mini batch sample
        indices = np.random.choice(self.memory_capacity, size = BATCH_SIZE)
        batch_trans = self.memory[indices, :]

        state = torch.FloatTensor(batch_trans[:, :self.s_dim])
        action = torch.FloatTensor(batch_trans[:, self.s_dim:self.s_dim + self.a_dim])
        reward = torch.FloatTensor(batch_trans[:, -self.s_dim - 1: -self.s_dim])
        next_state = torch.FloatTensor(batch_trans[:, -self.s_dim:])

        with torch.no_grad():
            noise = (torch.randn_like(action) * 0.2).clamp(-0.5, 0.5)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (self.gamma * target_Q).detach()

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.scheduler_critic.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

            self.scheduler_actor.step()

    def save(self, filename):
        # 保存模型
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        # 加载模型
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)














































