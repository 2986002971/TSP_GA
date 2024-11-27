import itertools
import time

import numpy as np
from data.medium_data import simplified_matrix


def solve_tsp_bruteforce(distance_matrix):
    """
    使用暴力法求解TSP问题

    Args:
        distance_matrix: 城市间距离矩阵

    Returns:
        best_path: 最短路径
        min_distance: 最短距离
    """
    n_cities = len(distance_matrix)
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
            curr_city = path[i]
            next_city = path[(i + 1) % n_cities]
            distance += distance_matrix[curr_city][next_city]

        # 更新最短距离和最佳路径
        if distance < min_distance:
            min_distance = distance
            best_path = path

    return np.array(best_path), min_distance


if __name__ == "__main__":
    start_time = time.time()

    # 使用相同的距离矩阵求解
    best_path, min_distance = solve_tsp_bruteforce(simplified_matrix)

    end_time = time.time()
    print(f"运行时间: {end_time - start_time:.2f}秒")

    print(f"最短路径: {best_path}")
    print(f"最短距离: {min_distance}")
