import itertools
import time

import numpy as np
from data.medium_data import simplified_data


def calculate_distance(city1, city2):
    """计算两个城市之间的欧氏距离"""
    dx = city1[0] - city2[0]
    dy = city1[1] - city2[1]
    return np.sqrt(dx * dx + dy * dy)


def solve_tsp_bruteforce_coords(coordinates):
    """
    使用暴力法求解TSP问题（直接使用坐标）

    Args:
        coordinates: 城市坐标列表 [[x1,y1], [x2,y2], ...]

    Returns:
        best_path: 最短路径
        min_distance: 最短距离
    """
    n_cities = len(coordinates)
    cities = list(range(n_cities))
    print(f"城市数量: {n_cities}")

    # 生成所有可能的路径
    all_paths = list(itertools.permutations(cities))
    print("路径组生成完毕")

    min_distance = float("inf")
    best_path = None

    # 计算每条路径的总距离
    for path in all_paths:
        distance = 0
        # 计算路径上相邻城市之间的距离，包括返回起点
        for i in range(n_cities):
            curr_city = coordinates[path[i]]
            next_city = coordinates[path[(i + 1) % n_cities]]
            distance += calculate_distance(curr_city, next_city)

        # 更新最短距离和最佳路径
        if distance < min_distance:
            min_distance = distance
            best_path = path

    return np.array(best_path), min_distance


if __name__ == "__main__":
    start_time = time.time()

    # 使用原始坐标数据求解
    best_path, min_distance = solve_tsp_bruteforce_coords(simplified_data)

    end_time = time.time()
    print(f"运行时间: {end_time - start_time:.2f}秒")

    print(f"最短路径: {best_path}")
    print(f"最短距离: {min_distance}")
