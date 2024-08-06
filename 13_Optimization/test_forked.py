import numpy as np
import time 
from pathlib import Path
import pandas as pd
from network_forked import AdjWeightGenerator, RoadNetwork
import pickle
from tqdm import tqdm
import datetime
"""
We test how adding forks to the networks increases solution time.
"""


# Set directories
date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
working_dir = Path().resolve().parent / "Documents" / "tobin_working_data" / "road_networks_forked"
output_dir = working_dir / f"tests_{date_time}"
working_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

# Create output dataframes
output_data = pd.DataFrame(columns=["size", "type", "adjacency_size", "actual_graph_rows", "actual_graph_columns", "time", "success"])

# Set the number of node sizes to be tested
sizes = [5, 7, 10, 12, 15, 17, 20, 22, 25, 30, 33, 35, 40]

# Set the number of iterations for each size
iterations = 10

def pickle_adjacency(adjacency, output_dir, size, iter):
    """
    Pickles the adjacency matrix.
    """
    with open(output_dir / f"adjacency_{size}_{iter}.pkl", "wb") as f:
        pickle.dump(adjacency, f)

for size in tqdm(sizes):
    for iter in tqdm(range(iterations)):
        r = RoadNetwork(size)

        # Run for unforked
        r.generate_unforked_adjacency()
        adj_size = r.adjacency.shape[0]
        r.load_graph()
        graph_r, graph_c = np.shape(r.g.graph)
        start_time = time.time()
        success = r.find_shortest_path()
        end_time = time.time()
        tot_time = end_time - start_time
        output_data = pd.concat([output_data, pd.DataFrame({"size": size, "type": "no_forks", "adjacency_size": adj_size, "actual_graph_rows":graph_r, "actual_graph_columns":graph_c, "time": tot_time, "success": success}, index=[0])], axis = 0).reset_index(drop = True)

        # Pickle and save CSV
        # output_data.to_csv(output_dir / "output_data.csv")
        # pickle_adjacency(r.adjacency, output_dir, size, iter)

        # Run for forked
        r.add_forks()
        r.activate_forks_random()
        adj_size = r.adjacency.shape[0]
        r.load_graph()
        graph_r, graph_c = np.shape(r.g.graph)
        start_time = time.time()
        success = r.find_shortest_path()
        end_time = time.time() 
        tot_time = end_time - start_time
        output_data = pd.concat([output_data, pd.DataFrame({"size": size, "type": "forks", "adjacency_size": adj_size, "actual_graph_rows":graph_r, "actual_graph_columns":graph_c,  "time": tot_time, "success": success}, index = [0])], axis = 0).reset_index(drop = True)

        # Pickle and save CSV
        # pickle_adjacency(r.adjacency, output_dir, size, iter)
        output_data.to_csv(output_dir / "output_data.csv")