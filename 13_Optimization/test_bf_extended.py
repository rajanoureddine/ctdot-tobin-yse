from network import *
import pandas as pd
from time import time
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pickle

# Set working directories
str_cwd = Path().resolve().parent
str_dir = str_cwd / "Documents" / "tobin_working_data" / "algo_tests"
str_weights = str_dir / "weights_adjacencies"

# Create an output dataframe
output = pd.DataFrame()

# Set graph sizes and constraints. Note that the actual number of nodes is size**2
sizes = [5, 10, 15, 20, 25]

# Set number of iterations for each size
iterations = 10

# Loop through the sizes
for size in tqdm(sizes):
    # Set the source and destination for the test, the destination is the last node
    src = 0
    dest = size**2-1

    # Iterate
    for i in tqdm(range(iterations)):
        # Generate the graph, and add weights
        generator = AdjWeightGenerator(size)
        generator.generate_adjacency(connection_probability=0.7)

        # Generate the weights and make charging possible
        generator.generate_weights(charging =True)
        g = generator.load_graph(size) # Since the constraint is the same as the size, we can use the size as the constraint
        start_time = time()
        g.BellmanExtended(src, dest, print_result = False)
        end_time = time()
        time_taken = end_time - start_time
        log_time = np.log(time_taken)
        success = len(g.path[dest]) > 0

        # Pickle the weight matrix and adjacency matrix
        with open(str_weights / f"weight_matrix_{size}_{i}.pkl", "wb") as f:
            pickle.dump(generator.weight_matrix, f)
        with open(str_weights / f"adjacency_{size}_{i}.pkl", "wb") as f:
            pickle.dump(generator.adjacency, f)

        # Create a DF row and append it to the output
        result = pd.DataFrame({"Size": size, "Log_N": np.log(size), "Constraint": size, "Iteration": i, "Time": time_taken, "Log_Time":log_time, "Success": success}, index = [0])
        output = pd.concat([output, result])

    # Save the output
    output.to_csv(str_dir / "bg_extended_tests.csv", index = False)

    

