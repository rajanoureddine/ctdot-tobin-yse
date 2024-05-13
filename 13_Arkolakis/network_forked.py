import numpy as np
from matplotlib import pyplot as plt
from bf_extended import GraphExtended
import pickle
import datetime
from pathlib import Path
from link import Link
import pandas as pd

class AdjWeightGenerator():
    def __init__(self, dim):
        self.size = dim            # Number of rows and columns. Num nodes = self.size ** 2

        # Create the adjacency and name columns and rows
        self.adjacency = pd.DataFrame(np.zeros((self.size**2, self.size**2)))
        self.adjacency.index = ["primary_node_" + str(i) for i in range(self.size**2)]
        self.adjacency.columns = ["primary_node_" + str(i) for i in range(self.size**2)]

        # Get the date and time as a string
        self.date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Set the working directory
        self.working_dir = Path().resolve().parent.parent / "Documents" / "tobin_working_data" / "road_networks_forked"
    
    def generate_adjacency(self):
        # Loop through the nodes, find their neighbours, and connect to some of them
        for i in range(self.size**2):
            for j in range(self.size**2):
                row_i = i // self.size
                col_i = i % self.size

                row_j = j // self.size
                col_j = j % self.size

                same_row = row_i == row_j
                same_col = col_i == col_j

                row_adjacent = (-1 <= (row_i - row_j)) and (1 >= (row_i - row_j))
                col_adjacent = (-1 <= (col_i - col_j)) and (1 >= (col_i - col_j))

                j_is_neighbour = (same_row and col_adjacent) or (same_col and row_adjacent)
                j_is_neighbour = j_is_neighbour and i != j

                # If the primary nodes are neighbours, and a joiner node has not been created, create one
                joiner_exists = ("joiner_node_" + str(i) + "_" + str(j) in self.adjacency.columns) or ("joiner_node_" + str(j) + "_" + str(i) in self.adjacency.columns)
                if j_is_neighbour and not joiner_exists:
                    self.adjacency.loc[:,"joiner_node_" + str(i) + "_" + str(j)] = 0
                    self.adjacency.loc["joiner_node_" + str(i) + "_" + str(j),:] = 0
                
                # If the primary nodes are neighbours, choose whether or not to link them (unidirectional link)
                if j_is_neighbour:
                    create_link = np.random.choice([0,1], 1, p=[0.5, 0.5])[0]
                    self.adjacency.loc["primary_node_" + str(i), "primary_node_" + str(j)] = create_link
                
                

        # Set all diagonal elements to 0
        for i in range(len(self.adjacency)):
            if self.adjacency.iloc[i,i] != 0:
                self.adjacency.loc[i,i] = 0
        
        return self.adjacency
    
    def generate_weights(self):
        for i in range(self.size**2):
            for j in range(self.size**2):
                try: # If the link exists
                    self.adjacency[i][j].set_main_weight(1)
                except: # If the link does not exist
                    pass
    
    def convert_to_std_adj_matrix(self):
        """
        Takes all of the dual forked links and turns them into a standard array (e.g.., if we have 10 nodes, we'll end up with many more).
        We take every possible link:
        A -------- B
        And make it
        x -------- C
        |          |
        A -------- B
        Thus the matrix
        0  0
        0  0
        Becomes
        0  0  0
        0  0  0 
        0  0  0
        For each two primary nodes, we have one additional node that can be forked to. 
        Thus if we have 2 primary nodes, we go from (2 * 2) --> (3 * 3). If we have 3 primary nodes, we go from (3 * 3) --> (5 * 5)
        Thus, if the original adjacency was (self.size**2 x self.size**2), 
        """
        out_adjacency = np.zeros((self.size**2, self.size**2))


                        


class RoadNetwork():
    def __init__(self, dim):
        self.size = dim            # Number of rows and columns. Num nodes = self.size ** 2

        # Create a generator object
        self.generator = AdjWeightGenerator(self.size)

        # Create the adjacency
        self.adjacency = [[0]*self.size**2]*self.size**2

        # Get the date and time as a string
        self.date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Set the working directory
        self.working_dir = Path().resolve().parent.parent / "Documents" / "tobin_working_data" / "road_networks_forked"

    def generate_adjacency(self):
        self.adjacency = self.generator.generate_adjacency()
    
    def generate_weights(self):
        self.generator.generate_weights()
    

    def plot_network(self, plot_path = False, path = None):
        """Given a valid adjacency matrix, plot the network"""

        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,10), facecolor = 'whitesmoke')

        ax.set_xlim((-1, self.size))
        ax.set_ylim((-1, self.size))

        # Iterate through each of the points and plot
        for i in range(self.size**2):
            for j in range(self.size**2):
                if self.adjacency[i][j]:  # There is a connection between node i, and node j
                    # Identify x, y, dx, dy for the arrow
                    x = i % self.size
                    y = i // self.size
                    dx = j % self.size - x
                    dy = j // self.size - y
                    x = x + dx/2
                    y = y + dy/2
                    dx = dx * 0.01
                    dy = dy * 0.01

                    # Identify line coords for line
                    x_vals = [i % self.size, j% self.size] 
                    y_vals = [i // self.size, j // self.size]

                    if self.adjacency[i][j]:
                        ax.plot(x_vals, y_vals, 'o', linestyle = '-', color = 'black', alpha =0.5, zorder = 10)
                        # ax.arrow(x, y, dx, dy, head_width = 0.1, head_length = 0.1, fc = 'black', ec = 'black', zorder = 5)
                    else:
                        ax.plot(x_vals, y_vals, linestyle = '-', color = 'green', linewidth = 4, zorder = 5)  
        
        if plot_path and path:
            for z in range(1, len(path)):
                i, j = path[z-1], path[z]
                x_vals = [i % self.size, j% self.size] 
                y_vals = [i // self.size, j // self.size]
                if self.weight_matrix[i,j] > 0:
                    ax.plot(x_vals, y_vals, 'o', linestyle = '-', color = 'red', linewidth = 3, zorder = 15)
                else:
                    ax.plot(x_vals, y_vals, 'o', linestyle = '-', color = 'purple', linewidth = 3, zorder = 15)
            fig.suptitle(f"Path: {','.join([str(x) for x in path])}", y = 0.90)
        elif plot_path and not path:
            fig.suptitle("No solution found", fontsize = 14, y = 0.90)

        plt.show()



if __name__ == '__main__':
    r = RoadNetwork(10)

    if False:
        date = "20240406_122037"
        # Load the adjacency matrix weights_20240406_120730.pkl using pickle
        with open(r.working_dir / f"adjacency_{date}.pkl", "rb") as f:
            adjacency = pickle.load(f)
        
        with open(r.working_dir / f"weights_{date}.pkl", "rb") as f:
            weights = pickle.load(f)

        # Set these as the adjacency and weight matrices
        r.adjacency = adjacency
        r.weight_matrix = weights

    if True:
        r.generate_adjacency()
        r.generate_weights()
        r.plot_network()

    if True:
        # Pickle the adjacency matrix
        with open(r.working_dir / f"adjacency_{r.date_time}.pkl", "wb") as f:
            pickle.dump(r.adjacency, f)
        
    
