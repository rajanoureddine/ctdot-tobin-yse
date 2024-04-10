import numpy as np
from matplotlib import pyplot as plt
from bf_extended import GraphExtended
import pickle
import datetime
from pathlib import Path

class AdjWeightGenerator():
    def __init__(self, dim):
        self.size = dim            # Number of rows and columns. Num nodes = self.size ** 2

        # Create the adjacency
        self.adjacency = np.zeros((self.size**2, self.size**2))
        self.weight_matrix = np.ones((self.size**2, self.size**2))

        # Get the date and time as a string
        self.date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Set the working directory
        self.working_dir = Path().resolve().parent / "Documents" / "tobin_working_data" / "road_networks"
    
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

                if j_is_neighbour:
                    self.adjacency[i, j] = np.random.choice([0,1], 1, p=[0.3, 0.7])[0]
        
        # Set all diagonal elements to 0
        for i in range(self.size**2):
            self.adjacency[i,i] = 0
        
        return self.adjacency
    
    def generate_weights(self, charging = False):
        for i in range(self.size**2):
            for j in range(self.size**2):
                if not charging:
                    self.weight_matrix[i,j] = 1
                elif charging:
                    self.weight_matrix[i,j] = np.random.choice([-1,1], 1, p = [0.3, 0.7])[0]
            
        return self.weight_matrix

    def load_graph(self, constraint):
        # Set constraint
        constraint = np.array([constraint]).reshape(-1)

        # Create the graph
        g = GraphExtended(self.size**2, 1, constraint)

        # Load the graph
        for i in range((self.size**2)):
            for j in range((self.size**2)):
                if self.adjacency[i,j] == 1:
                        if len(constraint) == 1:
                            g.addEdge(i,j, self.weight_matrix[i,j])
                        elif len(constraint) > 1:
                            g.addEdge(i,j, [1, self.weight_matrix[i,j]])
        
        # Return the graph
        return g
                        
        


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

    def generate_adjacency(self):
        self.adjacency = self.generator.generate_adjacency()
    
    def generate_weights(self, charging = False):
        self.weight_matrix = self.generator.generate_weights(charging = charging)

    def load_graph(self, constraint):
        """Loads the graph with the adjacency data.
        For now, assume each link has length 1"""

        self.g = self.generator.load_graph(constraint)


    def get_path(self, src, dest):
        self.g.BellmanExtended(src, dest, False)

        path = self.g.path[dest]

        if len(path)>0:
            solution = path[0].path
            self.plot_network(plot_path = True, path = solution)
        else:
            self.plot_network(plot_path = True)
        
        self.g.printPaths()
    

    def plot_network(self, plot_path = False, path = None):
        """Given a valid adjacency matrix, plot the network"""

        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,10), facecolor = 'whitesmoke')

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
        r.generate_weights(charging=False)
        r.plot_network()
        r.load_graph(10)
        r.get_path(0, 99)
        r.generate_weights(charging=True)

    r.plot_network()
    r.load_graph([np.infty, 10])
    r.get_path(0, 99)

    if True:
        # Pickle the adjacency matrix
        with open(r.working_dir / f"adjacency_{r.date_time}.pkl", "wb") as f:
            pickle.dump(r.adjacency, f)
        
        # Pickle the weight matrix
        with open(r.working_dir / f"weights_{r.date_time}.pkl", "wb") as f:
            pickle.dump(r.weight_matrix, f)
    
