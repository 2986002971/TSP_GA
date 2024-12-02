#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <float.h>
#include <queue>
#include <Python.h>

#define THREADS_PER_BLOCK 256
#define MAX_CITIES 32  // 考虑到使用int作为位掩码的限制

struct TSPState {
    int visited_mask;          // 已访问城市的位掩码
    float current_distance;    // 当前已经走过的距离
    int path[MAX_CITIES];      // 当前路径
    int visited_count;         // 已访问的城市数量
};

struct SharedMemoryManager {
    float distances[MAX_CITIES][MAX_CITIES];
    
    struct {
        float best_distance;
        int best_path[MAX_CITIES];
    } warp_results[THREADS_PER_BLOCK/32];
    
    float block_best_distance;
    int block_best_path[MAX_CITIES];
};

// 全局最优解
__device__ float global_best_distance;
__device__ int global_best_path[MAX_CITIES];

// 添加float类型的原子最小值操作
__device__ float atomicMinFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int expected;
    do {
        expected = old;
        old = atomicCAS(addr_as_int, expected,
            __float_as_int(min(__int_as_float(expected), value)));
    } while (expected != old);
    return __int_as_float(old);
}

__device__ void updateGlobalBest(float best_distance, int* best_path, int n) {
    // 使用一个额外的锁来确保路径更新的原子性
    __shared__ bool is_updating;
    if (threadIdx.x == 0) {
        is_updating = false;
    }
    __syncthreads();
    
    while (atomicCAS((int*)&is_updating, 0, 1) != 0);  // 获取锁
    
    float old_distance = atomicMinFloat(&global_best_distance, best_distance);
    if (best_distance < old_distance) {
        for (int i = 0; i < n; i++) {
            global_best_path[i] = best_path[i];  // 不需要atomicExch
        }
    }
    
    __threadfence();  // 确保所有写操作完成
    is_updating = false;  // 释放锁
}

// 初始化任务池的函数
TSPState* initializeTaskPool(const float* distance_matrix, int n, int expand_depth, int& num_states) {
    std::queue<TSPState> queue;
    
    // 创建初始状态（从城市0开始）
    TSPState initial_state = {0};
    initial_state.visited_mask = 1;  // 标记城市0为已访问
    initial_state.current_distance = 0;
    initial_state.path[0] = 0;
    initial_state.visited_count = 1;
    
    queue.push(initial_state);
    
    // 按照指定深度展开任务
    for (int depth = 0; depth < expand_depth; depth++) {
        int level_size = queue.size();
        
        for (int i = 0; i < level_size; i++) {
            TSPState current = queue.front();
            queue.pop();
            
            // 尝试访问每个未访问的城市
            for (int next_city = 1; next_city < n; next_city++) {
                if (!(current.visited_mask & (1 << next_city))) {
                    TSPState new_state = current;
                    new_state.visited_mask |= (1 << next_city);
                    new_state.current_distance += distance_matrix[new_state.path[new_state.visited_count - 1] * n + next_city];
                    new_state.path[new_state.visited_count] = next_city;
                    new_state.visited_count++;
                    queue.push(new_state);
                }
            }
        }
    }
    
    num_states = queue.size();
    
    TSPState* result = new TSPState[num_states];
    for (int i = 0; i < num_states; i++) {
        result[i] = queue.front();
        queue.pop();
    }
    
    return result;
}

__device__ void warpReduceBestPath(SharedMemoryManager* shared_mem, float local_best_distance, 
                                   int* local_best_path, int n, int lane_id, int warp_id, bool active) {
    uint32_t active_mask = __ballot_sync(0xffffffff, active);
    
    for (int offset = 16; offset > 0; offset /= 2) {
        int target_lane = lane_id + offset;
        bool other_active = (active_mask >> target_lane) & 1;
        
        float other_distance = __shfl_down_sync(0xffffffff, local_best_distance, offset);
        int other_path[MAX_CITIES];
        for (int i = 0; i < n; i++) {
            other_path[i] = __shfl_down_sync(0xffffffff, local_best_path[i], offset);
        }
        
        if (lane_id < offset && other_active && other_distance < local_best_distance) {
            local_best_distance = other_distance;
            for (int i = 0; i < n; i++) {
                local_best_path[i] = other_path[i];
            }
        }
    }
    
    if (lane_id == __ffs(active_mask) - 1) {
        shared_mem->warp_results[warp_id].best_distance = local_best_distance;
        for (int i = 0; i < n; i++) {
            shared_mem->warp_results[warp_id].best_path[i] = local_best_path[i];
        }
    }
}

__device__ void blockReduceBestPath(SharedMemoryManager* shared_mem, int n, bool active) {
    if (!active) return;
    
    float best_distance = FLT_MAX;
    int best_path[MAX_CITIES];
    
    for (int i = 0; i < blockDim.x/32; i++) {
        float warp_distance = shared_mem->warp_results[i].best_distance;
        if (warp_distance < best_distance) {
            best_distance = warp_distance;
            for (int j = 0; j < n; j++) {
                best_path[j] = shared_mem->warp_results[i].best_path[j];
            }
        }
    }
    
    updateGlobalBest(best_distance, best_path, n);
}

__global__ void solveTSP(const float* distance_matrix, const TSPState* initial_states, 
                         const int num_states, const int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    bool active = tid < num_states;

    __shared__ SharedMemoryManager shared_mem;
    
    // 初始化共享内存中的warp结果
    if (threadIdx.x < blockDim.x/32) {
        shared_mem.warp_results[threadIdx.x].best_distance = FLT_MAX;
    }
    
    // 加载距离矩阵到共享内存
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        for (int j = 0; j < n; j++) {
            shared_mem.distances[i][j] = distance_matrix[i * n + j];
        }
    }
    __syncthreads();
    
    if (!active) return;
    
    TSPState state = initial_states[tid];
    float local_best_distance = FLT_MAX;
    int local_best_path[MAX_CITIES];
    
    const int MAX_STACK_SIZE = MAX_CITIES * 2;
    TSPState stack[MAX_STACK_SIZE];
    int stack_top = 0;
    stack[stack_top++] = state;
    
    while (stack_top > 0) {
        state = stack[--stack_top];
        
        // 如果找到完整路径
        if (state.visited_count == n) {
            float return_distance = shared_mem.distances[state.path[n-1]][0];
            float total_distance = state.current_distance + return_distance;
            
            if (total_distance < local_best_distance) {
                local_best_distance = total_distance;
                for (int i = 0; i < n; i++) {
                    local_best_path[i] = state.path[i];
                }
            }
            continue;
        }
        
        // 尝试访问未访问的城市
        for (int next_city = 0; next_city < n; next_city++) {
            if (!(state.visited_mask & (1 << next_city))) {
                TSPState new_state = state;
                new_state.visited_mask |= (1 << next_city);
                float next_distance = shared_mem.distances[state.path[state.visited_count-1]][next_city];
                new_state.current_distance += next_distance;
                
                // 剪枝：如果新状态的距离已经超过了局部最优解，就不再入栈
                if (new_state.current_distance >= local_best_distance) {
                    continue;
                }
                
                new_state.path[state.visited_count] = next_city;
                new_state.visited_count++;
                
                // 检查栈是否溢出
                if (stack_top >= MAX_STACK_SIZE) {
                    printf("警告：栈溢出，当前栈深度：%d\n", stack_top);
                }
                
                stack[stack_top++] = new_state;
            }
        }
    }
    
    // 启动warp规约，所有活跃线程都会参与投票
    if (active) {
        warpReduceBestPath(&shared_mem, local_best_distance, local_best_path, n, lane_id, warp_id, active);
    }
    
    // 只有block中的第一个线程执行block规约
    if (threadIdx.x == 0) {
        blockReduceBestPath(&shared_mem, n, active);
    }
}

__global__ void initializeGlobals(int n) {
    global_best_distance = FLT_MAX;
    for (int i = 0; i < n; i++) {
        global_best_path[i] = -1;
    }
}

extern "C" void launchTSPSolver(const float* distance_matrix, int n, int expand_depth) {
    int num_states = 0;
    TSPState* initial_states = initializeTaskPool(distance_matrix, n, expand_depth, num_states);
    
    TSPState* d_initial_states;
    float* d_distance_matrix;
    
    cudaMalloc(&d_initial_states, num_states * sizeof(TSPState));
    cudaMalloc(&d_distance_matrix, n * n * sizeof(float));
    
    cudaMemcpy(d_initial_states, initial_states, num_states * sizeof(TSPState), cudaMemcpyHostToDevice);
    cudaMemcpy(d_distance_matrix, distance_matrix, n * n * sizeof(float), cudaMemcpyHostToDevice);
    
    initializeGlobals<<<1, 1>>>(n);
    cudaDeviceSynchronize();
    
    int blocks = (num_states + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    solveTSP<<<blocks, THREADS_PER_BLOCK>>>(d_distance_matrix, d_initial_states, num_states, n);
    
    cudaDeviceSynchronize();
    
    cudaFree(d_initial_states);
    cudaFree(d_distance_matrix);
    delete[] initial_states;
}

struct TSPResult {
    float best_distance;
    int best_path[MAX_CITIES];
};

extern "C" TSPResult* getTSPResult(int n) {
    TSPResult* result = new TSPResult;
    cudaMemcpyFromSymbol(&result->best_distance, global_best_distance, sizeof(float));
    cudaMemcpyFromSymbol(result->best_path, global_best_path, n * sizeof(int));
    return result;
}

static PyObject* py_solve_tsp(PyObject* self, PyObject* args) {
    PyObject* py_distance_matrix;
    int n, expand_depth;

    if (!PyArg_ParseTuple(args, "Oii", &py_distance_matrix, &n, &expand_depth)) {
        return NULL;
    }

    float* distance_matrix = new float[n * n];
    for (int i = 0; i < n * n; i++) {
        PyObject* value = PyList_GetItem(py_distance_matrix, i);
        distance_matrix[i] = (float)PyFloat_AsDouble(value);
    }

    launchTSPSolver(distance_matrix, n, expand_depth);
    
    TSPResult* result = getTSPResult(n);
    
    PyObject* result_dict = PyDict_New();
    PyDict_SetItemString(result_dict, "distance", PyFloat_FromDouble(result->best_distance));
    
    PyObject* path_list = PyList_New(n);
    for (int i = 0; i < n; i++) {
        PyList_SetItem(path_list, i, PyLong_FromLong(result->best_path[i]));
    }
    PyDict_SetItemString(result_dict, "path", path_list);

    delete[] distance_matrix;
    delete result;

    return result_dict;
}

static PyMethodDef TSPSolverMethods[] = {
    {"solve_tsp", py_solve_tsp, METH_VARARGS, "使用CUDA求解TSP问题"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef tspsolvemodule = {
    PyModuleDef_HEAD_INIT,
    "cuda_tsp_solver",
    "CUDA加速的TSP问题求解器",
    -1,
    TSPSolverMethods
};

PyMODINIT_FUNC PyInit_cuda_tsp_solver(void) {
    return PyModule_Create(&tspsolvemodule);
}
