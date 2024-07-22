import numpy as np
import random
import imageio.v2 as imageio

from multiprocessing import Pool, cpu_count

import tqdm
from os.path import join
from copy import deepcopy

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import cv2
import matplotlib.pyplot as plt

from robot_mazegame.Maze2 import Maze
from robot_mazegame.Robot_improve import Robot
from robot_mazegame.Runner2 import Runner

if __name__ == "__main__":
    # 实例化迷宫和机器人
    maze_output_file = "maze_structure.csv"
    traps_output_file = "maze_traps.csv"
    maze_c = Maze(maze_structure_file="maze_structure.csv",trap_position_file='maze_traps.csv')
    robot = Robot(maze_c)  # 创建机器人，传入迷宫实例

    # 设置机器人学习参数
    robot.set_status(learning=True, testing=False)  # 设置为学习模式

    # 实例化Runner
    runner = Runner(robot, maze_c)

    # 定义训练参数
    training_epoch = 100  # 训练轮数
    training_per_epoch = 150  # 每轮训练步数

    # 开始训练
    print("Starting training...")
    runner.run_training(training_epoch, training_per_epoch, display_direction=True)
    print("Training finished.")

    # 测试机器人
    print("Starting testing...")
    runner.run_testing(int(maze_c.height * maze_c.width * 0.85))  # 测试次数可以根据迷宫大小调整
    print("Testing finished.")

    # 生成视频
    runner.generate_movie("training_movie.avi")

    # 绘制结果
    runner.plot_results()