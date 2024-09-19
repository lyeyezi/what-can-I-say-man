import numpy as np
from tensorflow import keras
from collections import deque


class SnakeAI:
    def __init__(self, buffer_size=1000, batch_size=32):
        # 设置参数
        self.gamma = 0.99  # 折扣因子
        self.input_size = 12  # 输入状态的维度
        self.output_size = 4  # 输出动作的维度
        self.hidden_size = 100  # 隐藏层大小
        self.discount_factor = 0.99  # 训练目标的折扣因子

        # 创建神经网络模型
        self.model = self.build_model()  # 当前策略网络
        self.target_model = self.build_model()  # 目标策略网络
        self.model.compile(optimizer='adam', loss='mse')  # 编译模型
        self.target_model.compile(optimizer='adam', loss='mse')  # 编译目标模型

        # 经验回放缓冲区
        self.buffer = deque(maxlen=buffer_size)

        self.batch_size = batch_size

    def build_model(self):
        # 构建神经网络模型
        model = keras.Sequential()
        model.add(keras.layers.Dense(self.hidden_size, input_dim=self.input_size, activation='relu'))
        model.add(keras.layers.Dense(self.hidden_size, activation='relu'))
        model.add(keras.layers.Dense(self.hidden_size, activation='relu'))
        model.add(keras.layers.Dense(self.output_size, activation='linear'))
        return model

    def get_action(self, state):
        # 根据当前状态选择动作
        state = np.reshape(state, [1, self.input_size])
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def train_model(self):
        # 使用经验回放进行模型训练
        if len(self.buffer) < self.batch_size:
            return

        # 从经验回放缓冲区中随机采样一个批次的数据
        batch_indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[idx] for idx in batch_indices]

        # 解析批次数据
        states = np.array([sample[0] for sample in batch])
        actions = np.array([sample[1] for sample in batch])
        rewards = np.array([sample[2] for sample in batch])
        next_states = np.array([sample[3] for sample in batch])
        dones = np.array([sample[4] for sample in batch])

        # 计算训练目标
        targets = rewards + self.gamma * np.amax(self.model.predict_on_batch(next_states), axis=1) * (1 - dones)
        target_vec = self.model.predict_on_batch(states)
        indexes = np.array([i for i in range(self.batch_size)])
        target_vec[[indexes], [actions]] = targets

        # 使用批次数据训练模型
        self.model.fit(states, target_vec, epochs=1, verbose=0)

    def update_target_model(self):
        # 更新目标策略网络的权重
        self.target_model.set_weights(self.model.get_weights())

    def add_experience(self, state, action, reward, next_state, done):
        # 将经验添加到经验回放缓冲区中
        self.buffer.append((state, action, reward, next_state, done))
