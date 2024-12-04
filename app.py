import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from tsp.map_data import map_data
from tsp.tsp import TSPGeneticSolver

st.set_page_config(layout="wide")
st.title("TSP 遗传算法求解器")

# 侧边栏
st.sidebar.header("算法参数")
pop_size = st.sidebar.slider("种群大小", 50, 2000, 100)
generations = st.sidebar.slider("迭代次数", 100, 2000, 1000)
crossover_rate = st.sidebar.slider("交叉率", 0.0, 1.0, 0.8)
mutation_rate = st.sidebar.slider("变异率", 0.0, 1.0, 0.1)

if st.sidebar.button("开始优化"):
    # 创建求解器
    solver = TSPGeneticSolver(
        map_data,
        pop_size=pop_size,
        generations=generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
    )

    # 求解
    best_path, best_distance, history = solver.evolve()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("最优路径图")
        plt.figure(figsize=(8, 6))
        coords = np.array(map_data)
        plt.scatter(coords[:, 0], coords[:, 1], color="red", label="Cities")

        path_coords = coords[best_path]
        plt.plot(
            path_coords[:, 0],
            path_coords[:, 1],
            color="blue",
            linewidth=2,
            label="Optimal Path",
        )
        plt.scatter(
            path_coords[0, 0],
            path_coords[0, 1],
            color="green",
            s=100,
            label="Starting Point",
        )

        plt.title("Optimal Path")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        st.pyplot(plt)

        st.write(f"Shortest Distance: {best_distance:.2f}")

    with col2:
        st.subheader("优化过程")
        plt.figure(figsize=(8, 6))
        plt.plot(history, label="Shortest Distance")
        plt.title("Convergence Curve")
        plt.xlabel("Number of Generations")
        plt.ylabel("Path Length")
        plt.legend()
        st.pyplot(plt)

st.sidebar.markdown("""
### 说明
- 种群大小：同时维护的解决方案数量
- 迭代次数：算法运行的代数
- 交叉率：两个解决方案交换信息的概率
- 变异率：解决方案发生随机变化的概率
""")
