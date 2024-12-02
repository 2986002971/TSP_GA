import random
import time

import cuda_tsp_solver
from data.medium_data import simplified_matrix


def generate_simplified_matrix(size, greater_than_one_ratio):
    """
    生成一个简化的距离矩阵，主要距离为1，部分距离大于1。

    :param size: 矩阵的阶数
    :param greater_than_one_ratio: 大于1的距离的比例（0到1之间）
    :return: 生成的距离矩阵
    """
    matrix = [[1 if i != j else 0 for j in range(size)] for i in range(size)]

    # 计算需要大于1的距离的数量
    num_greater_than_one = int(size * (size - 1) * greater_than_one_ratio)

    # 随机选择位置并设置大于1的距离
    positions = set()
    while len(positions) < num_greater_than_one:
        i = random.randint(0, size - 1)
        j = random.randint(0, size - 1)
        if i != j:  # 确保不选择对角线
            positions.add((i, j))

    for i, j in positions:
        matrix[i][j] = random.randint(2, 10)  # 设置大于1的距离为2到10之间的随机值
        matrix[j][i] = matrix[i][j]

    return matrix


def test_cuda_tsp():
    # size = 12
    # greater_than_one_ratio = 0.2
    # simplified_matrix = generate_simplified_matrix(size, greater_than_one_ratio)

    # # 打印生成的矩阵
    # for row in simplified_matrix:
    #     print(row)

    # 将二维列表展平成一维列表
    one_dim_matrix = []
    for row in simplified_matrix:
        one_dim_matrix.extend(row)

    n = len(simplified_matrix)
    print(f"城市数量: {n}")

    print("\n开始CUDA计算...")
    start_time = time.time()

    expand_depth = 7
    result = cuda_tsp_solver.solve_tsp(one_dim_matrix, n, expand_depth)

    end_time = time.time()
    print(f"CUDA计算耗时: {end_time - start_time:.2f}秒")

    # 打印结果
    print(f"最短路径: {result['path']}")
    print(f"最短距离: {result['distance']}")


if __name__ == "__main__":
    test_cuda_tsp()
