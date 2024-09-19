class Snake:
    def __init__(self):
        # 初始化蛇的属性。
        pass

    def get_head_position(self):
        # 获取蛇头的位置。
        pass


def turn(self, point):
    # 改变蛇的移动方向。
    pass


def move(self):
    # 移动蛇的位置。
    pass


def reset(self):
    # 重新开始游戏。
    pass


def draw(self, surface):
    # 在窗口上绘制蛇。
    pass


class Food:
    def __init__(self):
        # 初始化食物的属性。
        pass

    def get_position(self):
        # 获取食物的位置。
        pass

    def draw(self, surface):
        # 在窗口上绘制食物。
        pass
import pygame
import sys
import random

# 定义常量
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
POP_SIZE = 10
BLOCK_SIZE = 20

# 定义颜色
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)


class Snake:
    # 初始化（）
    # 初始化蛇的长度（length），蛇的位置（positions），蛇的移动方向（direction），蛇的颜色（color = Green）
    def __init__(self):
        self.length = 3
        self.positions = [(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)]
        self.direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        self.color = GREEN

    # 获得蛇头的坐标（）
    def get_head_position(self):
        return self.positions[0]

    # 改变蛇移动方向（point：改变方向）
    #   如果改变的方向和蛇的原方向相反，则蛇的方向不改变
    #   否则，改变蛇的移动方向
    def turn(self, point):
        if (point[0] * -1, point[1] * -1) == self.direction:
            return
        else:
            self.direction = point

    # 移动（）
    #   根据当前方向计算下一个位置
    def move(self):
        cur = self.get_head_position()
        x, y = self.direction
        new = ((cur[0] + (x * BLOCK_SIZE)) % SCREEN_WIDTH, (cur[1] + (y * BLOCK_SIZE)) % SCREEN_HEIGHT)
        self.positions.insert(0, new)
        if len(self.positions) > self.length:
            self.positions.pop()

    # 重新开始（）
    def reset(self):
        self.length = 3
        self.positions = [(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)]
        self.direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])

    # 画蛇（surface：窗口对象）
    #   遍历蛇身的位置，将其画在画布上
    def draw(self, surface):
        for p in self.positions:
            r = pygame.Rect((p[0], p[1]), (BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(surface, self.color, r)
            pygame.draw.rect(surface, BLACK, r, 1)


class Food:
    # 初始化（）
    #   初始化食物的位置（position）和颜色（color = RED）
    def __init__(self):
        x = random.randrange(0, SCREEN_WIDTH, BLOCK_SIZE)
        y = random.randrange(0, SCREEN_HEIGHT, BLOCK_SIZE)
        self.position = (x, y)
        self.color = RED

    # 获得食物的坐标（）
    def get_position(self):
        return self.position

    # 画食物（surface：窗口对象）
    #   将食物画在画布上
    def draw(self, surface):
        r = pygame.Rect((self.position[0], self.position[1]), (BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(surface, self.color, r)
        pygame.draw.rect(surface, BLACK, r, 1)
