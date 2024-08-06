import numpy as np
from matplotlib import pyplot as plt
from dijkstra import Graph
import pickle
import datetime
from pathlib import Path

class AdjWeightGenerator():
    def __init__(self, dim):
        self.size = dim           

        self.validlinks = {0:10, 10:11, 11:21, 21:22, 22:32, 32:33, 33:43, 43:44, 44:54, 54:55, 55:65, 65:66, 66:76, 76:77, 77:87, 87:88, 88:98, 98:99}

        # Create the adjacency
        self.adjacency = np.zeros((self.size**2, self.size**2))
        self.weight_matrix = np.ones((self.size**2, self.size**2))

        # Get the date and time as a string
        self.date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Set the working directory
        self.working_dir = Path().resolve().parent / "Documents" / "tobin_working_data" / "road_networks"
    
    def generate_adjacency(self, diag = True, nondiag_sparse = False):
        # Loop through the nodes, find their neighbours, and connect to some of them
        for i in range(self.size**2):
            for j in range(self.size**2):
                if diag == False:
                    row_i = i // self.size
                    col_i = i % self.size

                    row_j = j // self.size
                    col_j = j % self.size

                    same_row = row_i == row_j
                    same_col = col_i == col_j

                    row_adjacent = (-1 <= (row_i - row_j)) and (1 >= (row_i - row_j))
                    col_adjacent = (-1 <= (col_i - col_j)) and (1 >= (col_i - col_j))

                    j_is_neighbour = (same_row and col_adjacent) or (same_col and row_adjacent)

                    if j_is_neighbour and not nondiag_sparse:
                        self.adjacency[i,j] = 1
                    elif j_is_neighbour and nondiag_sparse:
                        self.adjacency[i,j] = np.random.choice([0,1], p = [0.4, 0.6])
                else:
                    if self.validlinks.get(i) == j:
                        self.adjacency[i,j] = 1
                     
        # Set all diagonal elements to 0
        for i in range(self.size**2):
            self.adjacency[i,i] = 0
        
        return self.adjacency
                        

class RoadNetwork():
    def __init__(self, dim):
        self.size = dim            # Number of rows and columns. Num nodes = self.size ** 2

        # Create a generator object
        self.generator = AdjWeightGenerator(self.size)

        # Create the adjacency
        self.adjacency = np.zeros((self.size**2, self.size**2))
        self.weight_matrix = np.ones((self.size**2, self.size**2))

        # Get the date and time as a string
        self.date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Set the working directory
        self.working_dir = Path().resolve().parent / "Documents" / "tobin_working_data" / "road_networks"

    def generate_adjacency(self, diag, nondiag_sparse):
        self.adjacency = self.generator.generate_adjacency(diag, nondiag_sparse)
    
    def add_chargers(self):
        self.adjacency[11,12] = 1
        self.adjacency[12,22] = 1
        self.weight_matrix[12,22] = 4

        self.adjacency[44,45] = 1
        self.adjacency[45,55] = 1
        self.weight_matrix[45,55] = 4

    def add_more_chargers(self):
        self.adjacency[22,23] = 1
        self.adjacency[23,13]=1
        self.weight_matrix[23,13] = 4

        self.adjacency[5,6] = 1
        self.adjacency[6,16]=1
        self.weight_matrix[6,16] = 8

        self.adjacency[55,56] = 1
        self.adjacency[56,66] = 1
        self.weight_matrix[56,66] = 4

        self.adjacency[66,67]=1
        self.adjacency[67,77]=1
        self.weight_matrix[67,77] = 4

        self.adjacency[71,72] = 1
        self.adjacency[72,82] = 1
        self.weight_matrix[72,82] = 4


    def load_graph(self, constraint, objective_type = "nonlinear", alpha1 = 0.2, alpha2 = 0.8):
        """Loads the graph with the adjacency data.
        For now, assume each link has length 1"""

        self.objective_type = objective_type
        self.constraint = constraint

        self.alpha1 = alpha1
        self.alpha2 = alpha2

        # Initialize the graph
        self.g = Graph(self.size**2, constraint, objective_type)
        self.g.set_params(alpha1, alpha2)

        for i in range(self.size**2):
            for j in range(self.size**2):
                if self.adjacency[i,j] == 1 and self.weight_matrix[i,j] ==1:
                    self.g.addEdge(i, j, 1)
                elif self.adjacency[i,j] == 1:
                    self.g.addEdge(i, j, 2, self.weight_matrix[i,j])

    def get_path(self, src, dest):
        path, tc, c, b = self.g.Dijkstra(src, dest, print_it = True, return_path = True)
        self.path = path
        self.path_total_cost = tc
        self.path_cost = c
        self.path_boost = b
    

    def plot_network(self, plot_path = False, path = None):
        """Given a valid adjacency matrix, plot the network"""

        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8,8), facecolor = 'whitesmoke')

        ax.set_xlim((-1, self.size))
        ax.set_ylim((-1, self.size))

        # Iterate through each of the points and plot
        for i in range(self.size**2):
            for j in range(self.size**2):
                if self.adjacency[i,j] == 1:  # There is a connection between node i, and node j
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

                    if self.weight_matrix[i,j] == 1:
                        ax.plot(x_vals, y_vals, 'o', linestyle = '-', color = 'black', alpha =0.5, zorder = 10)
                    else:
                        ax.plot(x_vals, y_vals, linestyle = '-', color = 'green', linewidth = 4, zorder = 5)  
        
        if plot_path and path != "No path":
            for z in range(1, len(path)):
                i, j = path[z-1], path[z]

                x_vals = [i % self.size, j% self.size] 
                y_vals = [i // self.size, j // self.size]

                if True:
                    if self.weight_matrix[i,j] == 1:
                        ax.plot(x_vals, y_vals, 'o', linestyle = '-', color = 'red', linewidth = 3, zorder = 15)
                    else:
                        ax.plot(x_vals, y_vals, 'o', linestyle = '-', color = 'purple', linewidth = 3, zorder = 15)
                

                if False:
                    # Get the next link to see if it is a charging link
                    try:
                        i2, j2 = path[z], path[z+1]
                    except: # If we are at the last link, will throw an IndexError
                        i2, j2 = None, None

                    x_vals = [i % self.size, j% self.size] 
                    y_vals = [i // self.size, j // self.size]

                    # If this is not a charging link
                    if (self.weight_matrix[i,j] ==1):
                        # If it is the last link, plot it in red
                        if not i2 or not j2:
                            ax.plot(x_vals, y_vals, 'o', linestyle = '-', color = 'red', linewidth = 3, zorder = 15)
                        else:
                            # If the next link is not a charging link, plot the current link in red
                            if self.weight_matrix[i2,j2] == 1:
                                ax.plot(x_vals, y_vals, 'o', linestyle = '-', color = 'red', linewidth = 3, zorder = 15)
                            # If the next link is a charging link, plot the current link in purple
                            else: 
                                ax.plot(x_vals, y_vals, 'o', linestyle = '-', color = 'purple', linewidth = 3, zorder = 15)
                    # If the current link is a charging link, plot it in purple
                    else:
                        ax.plot(x_vals, y_vals, 'o', linestyle = '-', color = 'purple', linewidth = 3, zorder = 15)
            fig.suptitle(f"Path: {','.join([str(x) for x in path])}", y = 0.90)
            fig.text(0.5, 0.05, f"Total cost: {self.path_total_cost:.2f}, Cost: {self.path_cost}, Boost: {self.path_boost}", ha = 'center', fontsize = 12)
            if self.objective_type == "nonlinear":
                fig.text(0.5, 0.03, f"Total cost = ({self.alpha1} * {self.path_cost}) - {self.alpha2} * sqrt({self.constraint} - {self.path_cost} + {self.path_boost})", ha = 'center', fontsize = 12)
            else:
                fig.text(0.5, 0.03, f"Total cost = ({self.alpha1} * {self.path_cost}) - {self.alpha2} * ({self.constraint} - {self.path_cost} + {self.path_boost})", ha = 'center', fontsize = 12)
        elif plot_path and not path:
            fig.suptitle("No solution found", fontsize = 14, y = 0.90)

        plt.show()



if __name__ == '__main__':
    r = RoadNetwork(10)
    r.generate_adjacency(diag = False, nondiag_sparse = True)
    r.add_chargers()

    # Charges twice but not all four times
    if True:
        r.add_more_chargers()
        # r.plot_network()
    if True:
        r.load_graph(100, "nonlinear", 0.05, 0.95)
        r.get_path(0,99)
        r.plot_network(plot_path = True, path = r.path)



    
