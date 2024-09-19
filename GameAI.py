import pygame
import numpy as np

from game import Snake, Food, SCREEN_WIDTH, SCREEN_HEIGHT, BLOCK_SIZE, POP_SIZE, WHITE
from network import SnakeAI

# 初始化pygame
pygame.init()
font = pygame.font.SysFont('comics', 30)


class Game:
    # 初始化（）
    #   初始化屏幕大小（screen），刷新率（clock），蛇（snake），食物（food），要训练的ai（ai_player），
    #       载入已训练的模型（model.load_weights），成绩列表（scores），最好成绩（best_score），步数计数器（i）
    def __init__(self, buffer_size=1000, batch_size=32):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.snake = Snake()
        self.food = Food()
        self.ai_player = SnakeAI(buffer_size, batch_size)
      #  self.ai_player.model.load_weights('best_weights.h5') 有了在打开
        self.scores = []
        self.best_score = 0
        self.i = 0

    # 更新模型和蛇的行动（）
    #   根据当前状态state--model-->获得选择动作action，更新屏幕--奖励机制-->获得奖励，放入经验回放缓冲区--->使用经验回放训练模型
    def update(self, ai_player, tran_i):
        state = self.get_state()  # 获取当前状态
        action = ai_player.get_action(state)  # 根据当前状态选择动作
        v_a = self.snake.direction  # 记录原来的移动方向
        v_b = self.get_direction(action)  # 根据动作得到新的移动方向
        self.snake.turn(self.get_direction(action))  # 改变蛇的移动方向

        # 判断是否执行了无效转向和转向,若是则重置连续直线步数计数器
        if v_a != v_b and (v_a[0]*-1,v_a[1]*-1) != v_b:
            self.i = 0

        distances = np.sqrt(np.sum((np.array(self.snake.get_head_position()) - np.array(self.food.position)) ** 2))
        self.snake.move()  # 移动蛇的位置
        done = False

        if self.snake.get_head_position() == self.food.position:
            self.snake.length += 1
            self.food = Food()

        if self.is_collision():
            self.scores.append(self.snake.length)
            done = True

        next_state = self.get_state()  # 获取下一个状态
        reward = self.get_reward(done, distances, tran_i)  # 根据游戏情况计算奖励
        ai_player.add_experience(state, action, reward, next_state, done)  # 将经验添加到经验回放缓冲区
        ai_player.train_model()  # 使用经验回放训练模型
        return done

    def get_direction(self, action):
        # 根据动作索引获取移动方向
        if action == 0:
            return 0, -1
        elif action == 1:
            return 0, 1
        elif action == 2:
            return -1, 0
        else:
            return 1, 0

    def is_collision(self):
        # 判断蛇是否发生碰撞（头部位置是否与身体的其他部分重叠）
        return self.snake.get_head_position() in self.snake.positions[1:]

    def get_reward(self, done, distances, tran_i):
        distances_2 = np.sqrt(np.sum((np.array(self.snake.get_head_position()) - np.array(self.food.position)) ** 2))
        reward = 0
        if tran_i > 50:
            reward -= 0.1  # 连续直线20步之后，给与较小的惩罚（）
        if done:
            reward -= 20  # 如果碰撞到自己的身体，则给一个大的负向奖励
        elif self.snake.get_head_position() == self.food.position:
            reward += 10  # 如果蛇吃到食物，则给予一个较大的正向奖励
        elif distances_2 < distances:
            reward += 0.2  # 鼓励蛇靠近食物，给与正向奖励
        else:
            reward -= 0.1   # 惩罚蛇
        return reward

    def get_state(self):
        head = self.snake.get_head_position()
        food = self.food.position

        left = (head[0] - BLOCK_SIZE, head[1])
        right = (head[0] + BLOCK_SIZE, head[1])
        up = (head[0], head[1] - BLOCK_SIZE)
        down = (head[0], head[1] + BLOCK_SIZE)

        state = [
            # 朝左方向是否有障碍物
            (left in self.snake.positions[1:]),
            # 朝右方向是否有障碍物
            (right in self.snake.positions[1:]),
            # 朝上方向是否有障碍物
            (up in self.snake.positions[1:]),
            # 朝下方向是否有障碍物
            (down in self.snake.positions[1:]),

            # 食物是否在蛇的左侧
            food[0] < head[0],
            # 食物是否在蛇的右侧
            food[0] > head[0],
            # 食物是否在蛇的上方
            food[1] < head[1],
            # 食物是否在蛇的下方
            food[1] > head[1],

            # 蛇的朝向是否朝左
            self.snake.direction == (0, -1),
            # 蛇的朝向是否朝右
            self.snake.direction == (0, 1),
            # 蛇的朝向是否朝上
            self.snake.direction == (-1, 0),
            # 蛇的朝向是否朝下
            self.snake.direction == (1, 0),
        ]

        return np.asarray(state, dtype=np.float32)

    # 主流程（）
    #   初始化各个参数，进入游戏循环，直至游戏结束，实时保存模型参数
    def run(self):
        for _ in range(POP_SIZE):
            self.snake.reset()
            self.food = Food()
            done = False
            score = 0
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
                self.i += 1
                done = self.update(ai_player=self.ai_player, tran_i=self.i)
                if done:
                    break
                score = self.snake.length
                self.screen.fill(WHITE)
                self.snake.draw(self.screen)
                self.food.draw(self.screen)
                pygame.display.update()
                self.clock.tick(10000)

            self.ai_player.model.save_weights('best_weights.h5')
            self.best_score = score


game = Game(buffer_size=10000, batch_size=64)
game.run()
