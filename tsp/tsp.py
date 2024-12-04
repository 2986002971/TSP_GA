import numpy as np
from numba import float32, int32, jit

N_CITIES = 34  # 全局常量


@jit(float32[:, :](float32[:, :]), nopython=True)
def calculate_distance_matrix(coordinates):
    n = len(coordinates)
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            # 欧式距离
            dist = np.sqrt(
                (coordinates[i][0] - coordinates[j][0]) ** 2
                + (coordinates[i][1] - coordinates[j][1]) ** 2
            )
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist
    return dist_matrix


@jit(float32(int32[:], float32[:, :]), nopython=True)
def calculate_path_length(path, dist_matrix):
    total_distance = 0
    for i in range(N_CITIES - 1):
        total_distance += dist_matrix[path[i]][path[i + 1]]
    # 返回起点的距离
    total_distance += dist_matrix[path[N_CITIES - 1]][path[0]]
    return total_distance


@jit(int32[:](int32[:], int32[:]), nopython=True)
def ox_crossover(parent1, parent2):
    # 创建子代
    child = np.full(N_CITIES, -1, dtype=np.int32)

    # 选择一个子序列
    start = np.random.randint(0, N_CITIES - 2)
    end = np.random.randint(start + 1, N_CITIES - 1)

    # 1. 复制parent1的选定段到child
    child[start : end + 1] = parent1[start : end + 1]

    # 2. 从parent2中按顺序填充剩余位置
    current_idx = (end + 1) % N_CITIES
    parent2_idx = (end + 1) % N_CITIES

    while current_idx != start:
        # 找到parent2中下一个不在child中的城市
        while parent2[parent2_idx] in child:
            parent2_idx = (parent2_idx + 1) % N_CITIES

        child[current_idx] = parent2[parent2_idx]
        current_idx = (current_idx + 1) % N_CITIES
        parent2_idx = (parent2_idx + 1) % N_CITIES

    return child


@jit(int32[:](int32[:], float32), nopython=True)
def swap_mutation(individual, mutation_rate):
    if np.random.random() < mutation_rate:
        i, j = np.random.randint(0, N_CITIES, 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual


@jit(int32(float32[:]), nopython=True)
def tournament_select(fitness):
    # 随机选择k个个体进行锦标赛
    k = 3  # 锦标赛大小
    selected_idx = np.random.randint(0, len(fitness), k)
    # 返回适应度最高的个体的索引
    return selected_idx[np.argmax(fitness[selected_idx])]


@jit(float32[:](int32[:, :], float32[:, :]), nopython=True)
def calculate_population_fitness(population, dist_matrix):
    fitness = np.zeros(len(population), dtype=np.float32)
    for i in range(len(population)):
        # 使用路径长度的倒数作为适应度，路径越短适应度越高
        distance = calculate_path_length(population[i], dist_matrix)
        fitness[i] = 1.0 / distance
    return fitness


class TSPGeneticSolver:
    def __init__(
        self,
        coordinates,
        pop_size=100,
        generations=1000,
        crossover_rate=0.8,
        mutation_rate=0.1,
    ):
        self.coordinates = np.array(coordinates, dtype=np.float32)
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.n_cities = len(coordinates)

        # 预计算距离矩阵
        self.dist_matrix = calculate_distance_matrix(self.coordinates)

        # 初始化种群
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            # 随机生成一个路径
            path = np.random.permutation(self.n_cities)
            population.append(path)
        return np.array(population)

    def evolve(self):
        # 将coordinates和dist_matrix转换为float32类型
        self.coordinates = self.coordinates.astype(np.float32)
        self.dist_matrix = self.dist_matrix.astype(np.float32)

        # 将population转换为int32类型
        self.population = self.population.astype(np.int32)

        best_distance = float("inf")
        best_path = None
        # 记录每代的最佳距离，用于可视化
        history_distances = np.zeros(self.generations, dtype=np.float32)

        for generation in range(self.generations):
            fitness = calculate_population_fitness(self.population, self.dist_matrix)

            best_idx = np.argmax(fitness)
            current_distance = 1.0 / fitness[best_idx]
            history_distances[generation] = current_distance

            if current_distance < best_distance:
                best_distance = current_distance
                best_path = self.population[best_idx].copy()

            # 创建新一代
            new_population = np.empty((self.pop_size, N_CITIES), dtype=np.int32)

            # 生成新的个体
            for i in range(self.pop_size):
                if np.random.random() < self.crossover_rate:
                    # 选择亲代
                    parent1_idx = tournament_select(fitness)
                    parent2_idx = tournament_select(fitness)
                    # 交叉
                    child = ox_crossover(
                        self.population[parent1_idx], self.population[parent2_idx]
                    )
                else:
                    # 直接复制
                    child = self.population[tournament_select(fitness)].copy()

                # 变异
                child = swap_mutation(child, self.mutation_rate)
                new_population[i] = child

            self.population = new_population

        return best_path, best_distance, history_distances
