import itertools

import torch
import triton
import triton.language as tl
from map_data import simplified_matrix


@triton.jit
def evaluate_paths_kernel(
    dist_matrix_ptr,  # 距离矩阵指针 [n_cities, n_cities]
    paths_ptr,  # 所有路径数组指针 [n_paths, n_cities]
    n_cities: tl.constexpr,  # 城市数量
    next_power_of_2: tl.constexpr,  # 城市数量向上取整的2的幂
    n_paths: tl.constexpr,  # 总路径数量
    output_ptr,  # 输出指针 [n_paths, 1]
    best_dist_ptr,  # 输出最短距离指针 [1]
    best_path_ptr,  # 输出最佳路径指针 [n_cities]
    BLOCK_SIZE: tl.constexpr,
):
    """
    评估所有路径的距离，找出最短路径
    """
    # 每个线程块处理一组路径
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_paths

    # 计算当前一组路径的起点在paths_ptr中的位置
    path_start = offsets * n_cities

    # 计算城市索引，由于arange只能生成2的幂次，所以需要计算向上取整的2的幂次，然后截取到n_cities
    city_offsets = tl.arange(0, next_power_of_2)
    city_mask = city_offsets < n_cities
    """
    利用广播机制，一次性加载一组路径的所有城市索引
    假设
    BLOCK_SIZE = 2
    n_cities = 3

    则可知
    path_start = [0, 3]

    则
    city_offsets = [0, 1, 2]
    path_start[:, None] = [[0],
                        [3]]
    city_offsets[None, :] = [[0, 1, 2]]

    加载当前城市的索引，广播运算：
    [[0, 1, 2],
    [3, 4, 5]]

    加载下一个城市的索引（模运算）：
    [[1, 2, 0],
    [4, 5, 3]]
    """
    curr_cities = tl.load(
        paths_ptr + path_start[:, None] + city_offsets[None, :],
        mask=mask[:, None] & city_mask[None, :],
    )
    next_cities = tl.load(
        paths_ptr + path_start[:, None] + ((city_offsets + 1) % n_cities)[None, :],
        mask=mask[:, None] & city_mask[None, :],
    )

    # 加载每对相邻城市之间的距离，curr_cities访问行索引，next_cities访问列索引
    dists = tl.load(
        dist_matrix_ptr + curr_cities * n_cities + next_cities, mask=mask[:, None]
    )
    # 对每条路径的所有段距离求和
    distances = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    distances = tl.sum(dists, axis=1)

    # 存储每条路径的距离
    tl.store(output_ptr + offsets, distances, mask=mask)

    # 找出最短距离和对应的路径
    min_dist = tl.min(distances, axis=0)
    min_idx = tl.argmin(distances, axis=0)

    # 只有第一个线程写入结果
    if pid == 0:
        tl.store(best_dist_ptr, min_dist)
        best_path_start = path_start[min_idx]
        path_vals = tl.load(paths_ptr + best_path_start + city_offsets)
        tl.store(best_path_ptr + city_offsets, path_vals)


def find_shortest_path(paths):
    dist_matrix = torch.tensor(simplified_matrix, dtype=torch.float32, device="cuda")
    paths_tensor = torch.tensor(paths, dtype=torch.int32, device="cuda")
    n_cities = len(paths[0])
    next_power_of_2 = 1 << (n_cities - 1).bit_length()
    n_paths = len(paths)
    output_ptr = torch.zeros(n_paths, dtype=torch.float32, device="cuda")
    best_distance = torch.zeros(1, dtype=torch.float32, device="cuda")
    best_path = torch.zeros(n_cities, dtype=torch.int32, device="cuda")

    BLOCK_SIZE = 32
    grid = (triton.cdiv(n_paths, BLOCK_SIZE),)

    evaluate_paths_kernel[grid](
        dist_matrix_ptr=dist_matrix.data_ptr(),
        paths_ptr=paths_tensor.data_ptr(),
        n_cities=n_cities,
        next_power_of_2=next_power_of_2,
        n_paths=n_paths,
        output_ptr=output_ptr.data_ptr(),
        best_dist_ptr=best_distance.data_ptr(),
        best_path_ptr=best_path.data_ptr(),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return best_distance.item(), best_path.cpu().numpy()


if __name__ == "__main__":
    # 生成所有可能的路径排列（对于10个城市）
    n_cities = 10
    city_indices = list(range(n_cities))
    all_paths = list(itertools.permutations(city_indices))

    # 找出最短路径
    min_distance, best_path = find_shortest_path(all_paths)
    print(f"最短路径: {best_path}")
    print(f"最短距离: {min_distance}")
