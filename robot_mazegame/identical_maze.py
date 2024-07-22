# 导入必要的库和Maze类
from robot_mazegame.Maze2 import Maze  # 假设Maze类定义在your_module.py文件中
import numpy as np
import matplotlib.pyplot as plt

# 实例化Maze对象，这里以随机生成为例
# maze_size = (15, 15)  # 定义迷宫的大小
# my_maze = Maze(maze_size=maze_size)

# 将生成的迷宫数据保存到CSV文件
maze_output_file = "maze_structure.csv"
# np.savetxt(maze_output_file, my_maze.maze_data, fmt='%d', delimiter=',')
# print(f"迷宫数据已保存至：{maze_output_file}")

# 将生成的陷阱位置保存到CSV文件
traps_output_file = "maze_traps.csv"
# np.savetxt(traps_output_file, my_maze._traps_array, fmt='%d', delimiter=',')
# print(f"陷阱位置已保存至：{traps_output_file}")

loaded_maze = Maze(maze_structure_file="maze_structure.csv",trap_position_file='maze_traps.csv')

plt.figure(figsize=(loaded_maze.height, loaded_maze.width))
plt.imshow(loaded_maze.draw_current_maze())
plt.axis('off')
plt.show()