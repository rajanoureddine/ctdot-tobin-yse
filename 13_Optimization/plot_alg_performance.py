import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from timeit import Timer
import time
from bellman_ford import Graph, Path, GraphExtended

# Create a function to test the performance of the Bellman-Ford algorithm
def setup_graph(size):
    graph_size = size
    graph = np.zeros([graph_size, graph_size])
    for i in range(graph_size):
        for j in range(graph_size):
            graph[i][j] = np.random.choice([0, 1], p=[0.7, 0.3])
    graph_weights = np.random.randint(1, 20, (graph_size, graph_size))
    g = Graph(graph_size)
    for i in range(graph_size):
        for j in range(graph_size):
            if graph[i][j] != 0:
                g.addEdge(i, j, graph_weights[i][j])
    return g, graph, graph_weights

def compare_performance(size, iterations):
    # Create a DataFrame to store the results
    results = pd.DataFrame(columns=['Graph Size', 'Time', 'Algorithm'])

    s = """"""

    # List
    times = {"Size": [size]* iterations, "Bellman-Ford": [], "Dijkstra": []}

    for i in range(iterations):
        g, graph, graph_weights = setup_graph(size)
        start_time = time.time()
        g.BellmanFord(0)
        bf_time = time.time() - start_time
        g.reset()
        start_time = time.time()
        g.Dijkstra(0)
        dijkstra_time = time.time() - start_time
        times["Bellman-Ford"].append(bf_time)
        times["Dijkstra"].append(dijkstra_time)
    
    return times

out_df = pd.DataFrame([])

for i in range(40, 201, 20):
    times = compare_performance(i, 4)
    df = pd.DataFrame(times)
    out_df = pd.concat([out_df, df]).reset_index(drop=True)
    print(f"Finished size {i}")

out_df.to_csv("algo_performance.csv", index=False)

# Group by size and get an average and standard deviation
grouped = out_df.groupby("Size").agg(["mean", "std"])
grouped.to_csv("algo_performance_grouped.csv")

# Plot the results
fig, ax = plt.subplots()
ax.plot(grouped.index, grouped["Bellman-Ford"]["mean"], label="Bellman-Ford")
ax.plot(grouped.index, grouped["Dijkstra"]["mean"], label="Dijkstra")
ax.fill_between(grouped.index, grouped["Bellman-Ford"]["mean"] - grouped["Bellman-Ford"]["std"], grouped["Bellman-Ford"]["mean"] + grouped["Bellman-Ford"]["std"], alpha=0.2)
ax.fill_between(grouped.index, grouped["Dijkstra"]["mean"] - grouped["Dijkstra"]["std"], grouped["Dijkstra"]["mean"] + grouped["Dijkstra"]["std"], alpha=0.2)
plt.xlabel("Graph Size")
plt.ylabel("Time (s)")
plt.title("Bellman-Ford vs Dijkstra Algorithm Performance")
plt.legend()
plt.show()





    
    
    