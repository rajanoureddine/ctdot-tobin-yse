from network import *
from scipy.sparse import csr_matrix, csgraph
from scipy.sparse.csgraph import dijkstra, bellman_ford
import numpy as np
import datetime
import logging
import pickle
import pandas as pd
from tqdm import tqdm

# Set working directory
date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
working_dir = Path().resolve().parent / "Documents" / "tobin_working_data" / "road_networks"
output_dir = working_dir / "tests_nonlinear" / f"tests_{date}"
output_dir.mkdir(parents=True, exist_ok=True)

# Create a log file in the output directory
logging.basicConfig(filename = output_dir / f"test_log_{date}.log", level = logging.INFO)
logging.log(logging.INFO, f"Testing nonlinear objective function")

# Set the sizes for test
sizes = [5, 7, 10, 12, 15, 17, 20, 22, 25, 30, 40]
iterations = 10

# Create an output dataframe
output = pd.DataFrame(columns = ["iteration", "size", "dijkstra_time", "dijkstra_path", "dijkstra_dist", "bellman_ford_time", "bellman_ford_path", "bellman_ford_dist", "bf_neg_time", "bf_neg_path", "bf_neg_dist", "pos_links", "neg_links"])

def get_path(predecessors, dest, src = 0):
    path = []
    node = dest
    while node != src:
        path.append(node)
        node = predecessors[node]
        if node == -9999:
            return "No path"
    path.append(src)
    path.reverse()
    path = ','.join([str(x) for x in path])
    return path

count = 0
for size in tqdm(sizes):
    dest = size**2 - 1

    logging.info(f"Testing size {size}")
    for iteration in range(iterations):
        logging.info(f"Testing iteration {iteration}")
        # Create a road network and adjacency
        r = RoadNetwork(size)
        r.generate_adjacency()
        r.generate_weights(charging = False)
        adj = r.adjacency

        # Pickle the adjacency for replicability
        start = datetime.datetime.now()
        with open(output_dir / f"adj_{size}_{iteration}.pkl", "wb") as f:
            pickle.dump(adj, f)
        end = datetime.datetime.now()
        logging.info(f"Time to pickle adjacency of size {size} on iteration {iteration}: {end-start}")

        # Load the adjacency
        adj = csr_matrix(r.adjacency * 3)

        # Run it for dijkstra and bellman ford
        start = datetime.datetime.now()
        dist_matrix, predecessors, _ = dijkstra(adj, indices = 0, min_only = True, return_predecessors = True)
        end = datetime.datetime.now()
        path = get_path(predecessors, dest)
        new_row = pd.DataFrame({"iteration": iteration, "size": size, "dijkstra_time": (end-start).total_seconds(), "dijkstra_path": path, "dijkstra_dist": dist_matrix[dest]}, index = [count])
        count +=1

        # Run if for bellman ford
        start = datetime.datetime.now()
        dist_matrix, predecessors = bellman_ford(adj,indices = 0, return_predecessors = True)
        end = datetime.datetime.now()
        path = get_path(predecessors, dest)
        new_row["bellman_ford_time"] = (end-start).total_seconds()
        new_row["bellman_ford_path"] = path
        new_row["bellman_ford_dist"] = dist_matrix[dest]
        count +=1

        # Run with negative weights for bf
        adj = r.adjacency.copy()
        for i in range(len(adj)):
            for j in range(len(adj)):
                if adj[i,j] == 1:
                    link = np.random.choice([-1,3],1, p = [0.03, 0.97])[0]
                    adj[i,j] = link

        adj_obj = csr_matrix(adj)
        try:
            start = datetime.datetime.now()
            dist_matrix, predecessors = bellman_ford(adj_obj,indices = 0, return_predecessors = True)
            end = datetime.datetime.now()
        except:
            new_row["bf_neg_time"] = np.nan
            new_row["bf_neg_path"] = "Negative cycle"
            new_row["bf_neg_dist"] = np.nan
            new_row["pos_links"] = np.nan
            new_row["neg_links"] = np.nan
            output = pd.concat([output ,new_row], axis = 0)
            count += 1

            # Save the output
            logging.info(f"Output for size {size} on iteration {iteration}:\n{output}")
            output.to_csv(output_dir / f"output_{date}.csv", index = False)

            continue

        path = get_path(predecessors, dest)
        new_row["bf_neg_time"] = (end-start).total_seconds()
        new_row["bf_neg_path"] = path
        new_row["bf_neg_dist"] = dist_matrix[dest]
        new_row["pos_links"] = np.sum(adj > 0)
        new_row["neg_links"] = np.sum(adj < 0)
        output = pd.concat([output ,new_row], axis = 0)
        count += 1

        # Save the output
        logging.info(f"Output for size {size} on iteration {iteration}:\n{output}")
        output.to_csv(output_dir / f"output_{date}.csv", index = False)


    

