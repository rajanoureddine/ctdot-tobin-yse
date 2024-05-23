import numpy as np
from matplotlib import pyplot as plt
import pickle
import datetime
from pathlib import Path
from link import Link
import pandas as pd
from dijkstra import *
import time

# Silence warnings
import warnings
warnings.filterwarnings("ignore")

class AdjWeightGenerator():
    def __init__(self, dim):
        self.size = dim            # Number of rows and columns. Num nodes = self.size ** 2

        # Create blank neighbours
        self.neighbours = np.zeros((self.size**2, self.size**2))
        self.get_all_neighbours()

        # Create blank joiners
        self.joiner_exists = np.zeros((self.size**2, self.size**2))

        # Create the adjacency and name columns and rows
        self.adjacency = pd.DataFrame(np.zeros((self.size**2, self.size**2)))
        self.adjacency.index = ["primary_node_" + str(i) for i in range(self.size**2)]
        self.adjacency.columns = ["primary_node_" + str(i) for i in range(self.size**2)]

        # Get the date and time as a string
        self.date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Set the working directory
        self.working_dir = Path().resolve().parent.parent / "Documents" / "tobin_working_data" / "road_networks_forked"


    
    def is_neighbour(self, i, j):
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

        return j_is_neighbour

    def get_all_neighbours(self):
        for i in range(self.size**2):
            for j in range(self.size**2):
                if self.is_neighbour(i, j):
                    self.neighbours[i, j] = 1

    def create_joiner(self, i, j):
        """
        Creates a joiner node between two primary nodes i and j. 
        """
        self.adjacency.loc[:,"joiner_node_" + str(i) + "_" + str(j)] = 0
        self.adjacency.loc["joiner_node_" + str(i) + "_" + str(j),:] = 0

        self.joiner_exists[i, j] = 1
        self.joiner_exists[j, i] = 1

    #def joiner_exists(self, i, j):
    #    """
    #    Returns True if a joiner node exists between two primary nodes i and j.
     #   """
    #    return ("joiner_node_" + str(i) + "_" + str(j) in self.adjacency.columns) or ("joiner_node_" + str(j) + "_" + str(i) in self.adjacency.columns)
    
    def zero_diagonal(self):
        # Set all diagonal elements to 0
        for i in range(len(self.adjacency)):
            if self.adjacency.iloc[i,i] != 0:
                self.adjacency.loc[i,i] = 0
    
    def join_primary_nodes(self):
        """
        Used to create links between nodes in the network - with NO forks
        """
        for i in range(self.size**2):
            for j in range(self.size**2):
                if self.neighbours[i, j]:
                    self.adjacency.loc["primary_node_" + str(i), "primary_node_" + str(j)] = np.random.choice([0,1], 1, p=[0.3, 0.7])[0]
        
        # Set all diagonal elements to 0
        self.zero_diagonal()

        return self.adjacency
    
    def add_forks(self):
        """
        Generates an adjacency matrix to represent a road network. Also produces "joiner nodes" to allow for forked links.
        """
        joiner_time = 0

        joiner_cols = []

        # Loop through the nodes, find their neighbours, and connect to some of them
        for i in range(self.size**2):
            for j in range(self.size**2):

                # If the primary nodes are neighbours, and a joiner node has not been created, create one
                # j_is_neighbour = self.is_neighbour(i, j)
                if self.neighbours[i,j] and not self.joiner_exists[i, j]:
                    joiner_cols = joiner_cols + ["joiner_node_" + str(i) + "_" + str(j)]
                    self.joiner_exists[i, j] = 1
                    self.joiner_exists[j, i] = 1

       
        # Add blank joiner columns
        self.adjacency[joiner_cols] = 0

        # Add blank joiner rows
        joiner_rows = pd.DataFrame(np.zeros((len(joiner_cols), len(self.adjacency.columns))), columns = self.adjacency.columns)
        joiner_rows.index = joiner_cols
        self.adjacency = pd.concat([self.adjacency, joiner_rows])
                    
        # Set all diagonal elements to 0
        self.zero_diagonal()

        # Return
        return self.adjacency
    
    def activate_forks_random(self):
        """
        Randomly activates the forks in the network.
        """
        for i in range(self.size**2):
            for j in range(self.size**2):

                # To activate a fork, i and j must be joined neighbours, with a joiner node.
                activate = self.neighbours[i,j]
                activate = activate and self.adjacency.loc["primary_node_" + str(i), "primary_node_" + str(j)] == 1
                activate = activate and i < j
                activate = activate and np.random.choice([0,1], 1, p=[0.5, 0.5])[0]

                if activate:
                    self.adjacency.loc["primary_node_" + str(i), "joiner_node_" + str(i) + "_" + str(j)] = 1
                    self.adjacency.loc["joiner_node_" + str(i) + "_" + str(j), "primary_node_" + str(j)] = 1
        
        return self.adjacency

    def add_charging_links(self):
        """
        Adds charging links to the network. If we want to add a charging link, we join from i to the joiner node, and from the joiner node to j.
        """
        for i in range(self.size**2):
            for j in range(self.size**2):
                primary_node_i = "primary_node_" + str(i)
                primary_node_j = "primary_node_" + str(j)
                
                # Identify the joiner node - remember there is only one for each pair of primary nodes
                if i < j:
                    joiner_node = "joiner_node_" + str(i) + "_" + str(j)
                else:
                    joiner_node = "joiner_node_" + str(j) + "_" + str(i)

                # Only do this where the nodes are already linked
                if self.is_neighbour(i, j) and self.check_linked(i, j):
                    add_charging = np.random.choice([0,1], 1, p=[0.5, 0.5])[0]
                    if add_charging:
                        self.adjacency.loc[primary_node_i, joiner_node] = 1
                        self.adjacency.loc[joiner_node, primary_node_j] = 1

        return self.adjacency

    def check_linked(self, i, j):
        """
        Returns True if there is a link between primary nodes i and j.
        """
        primary_node_i = "primary_node_" + str(i)
        primary_node_j = "primary_node_" + str(j)

        return self.adjacency.loc[primary_node_i, primary_node_j] == 1
    
    def check_charging(self, i, j):
        """
        Returns True if there is a charging link between primary nodes i and j.
        """
        primary_node_i = "primary_node_" + str(i)
        primary_node_j = "primary_node_" + str(j)
        joiner_node = "joiner_node_" + str(i) + "_" + str(j)

        try:
            has_charging = (self.adjacency.loc[primary_node_i, joiner_node] == 1) and (self.adjacency.loc[joiner_node, primary_node_j] == 1)
        except:
            has_charging = False

        return has_charging
     
    
class RoadNetwork():
    def __init__(self, dim):
        self.size = dim            # Number of rows and columns. Num nodes = self.size ** 2
        self.finalnode = (self.size**2) - 1

        # Create a generator object
        self.generator = AdjWeightGenerator(self.size)

        # Get the date and time as a string
        self.date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Set the working directory
        self.working_dir = Path().resolve().parent / "Documents" / "tobin_working_data" / "road_networks_forked"

    def generate_unforked_adjacency(self):
        self.adjacency = self.generator.join_primary_nodes()

    def add_forks(self):
        self.adjacency = self.generator.add_forks()

    def activate_forks_random(self):
        self.adjacency = self.generator.activate_forks_random()
    
    def add_charging_links(self):
        self.adjacency = self.generator.add_charging_links()

    def load_graph(self):
        self.g = Graph(len(self.adjacency))
        for i in range(len(self.adjacency)):
            for j in range(len(self.adjacency)):
                if self.adjacency.iloc[i,j] == 1:
                    self.g.addEdge(i, j, 1)
        
    def find_shortest_path(self, src = 0):
        success = self.g.Dijkstra(src, self.finalnode, True)
        return success


    def plot_network(self, plot_path = False, path = None):
        """Given a valid adjacency matrix, plot the network"""

        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,10), facecolor = 'whitesmoke')

        ax.set_xlim((-1, self.size))
        ax.set_ylim((-1, self.size))

        # Plot the nodes
        columns = self.adjacency.columns.tolist()
        rows = self.adjacency.index.tolist()

        for i in rows:
            for j in columns:
                if ("primary" in i) and ("primary" in j):
                    if self.adjacency.loc[i,j] == 1:
                        i_index = int(i.split("_")[-1])
                        j_index = int(j.split("_")[-1])

                        # Check if also charging
                        charging_link = self.generator.check_charging(i_index, j_index)

                        # Identify x, y, dx, dy for the arrow
                        x = i_index % self.size
                        y = i_index // self.size
                        dx = j_index % self.size - x
                        dy = j_index // self.size - y
                        x = x + dx/2
                        y = y + dy/2
                        dx = dx * 0.01
                        dy = dy * 0.01
 
                        # Identify line coords for line
                        x_vals = [i_index % self.size, j_index % self.size] 
                        y_vals = [i_index // self.size, j_index // self.size]

                        # Plot
                        if not charging_link:
                            ax.plot(x_vals, y_vals, 'o', linestyle = '-', color = 'black', alpha =0.5, zorder = 10)
                        else:
                            ax.plot(x_vals, y_vals, 'o', linestyle = '-', color = 'green', alpha =0.5, zorder = 10)

        
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
    r = RoadNetwork(15)

    if True:
        r.generate_unforked_adjacency()
        r.load_graph()

        r.find_shortest_path()
        
        r.add_forks()

        r.activate_forks_random()

        r.load_graph()

        r.find_shortest_path()


    if True:
        # Pickle the adjacency matrix
        with open(r.working_dir / f"adjacency_{r.date_time}.pkl", "wb") as f:
            pickle.dump(r.adjacency, f)
        
    
