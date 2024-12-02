import numpy as np
from data.map_data import map_data


def calculate_distance_matrix(coordinates):
    """计算城市间距离矩阵"""
    n_cities = len(coordinates)
    distances = np.zeros((n_cities, n_cities))

    for i in range(n_cities):
        for j in range(i + 1, n_cities):
            dx = coordinates[i][0] - coordinates[j][0]
            dy = coordinates[i][1] - coordinates[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            distances[i][j] = dist
            distances[j][i] = dist

    return distances.tolist()


n_cities = 15
distance_matrix = calculate_distance_matrix(map_data[:n_cities])

# 将结果写入到一个新的Python文件
with open("medium_data.py", "w") as f:
    f.write("simplified_data = ")
    f.write(str(map_data[:n_cities]))
    f.write("\n")
    f.write("simplified_matrix = ")
    f.write(str(distance_matrix))
    f.write("\n")

print("距离矩阵已写入到 medium_data.py 文件中。")
