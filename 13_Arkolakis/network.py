import numpy as np
from matplotlib import pyplot as plt
from bf_extended import GraphExtended

class RoadNetwork():
    def __init__(self, dim):
        self.size = dim            # Number of rows and columns. Num nodes = self.size ** 2

        # Create the adjacency
        self.adjacency = np.zeros((self.size**2, self.size**2))
        self.weight_matrix = np.ones((self.size**2, self.size**2))
        

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
    
    def generate_weights(self, charging = False):
        for i in range(self.size**2):
            for j in range(self.size**2):
                if not charging:
                    self.weight_matrix[i,j] = 1
                if charging:
                    self.weight_matrix[i,j] = np.random.choice([-1,1], 1, p = [0.3, 0.7])[0]


    def load_graph(self, constraint):
        """Loads the graph with the adjacency data.
        For now, assume each link has length 1"""

        # Create the graph
        self.g = GraphExtended(self.size**2, 1, np.array([constraint]))

        for i in range((self.size**2)):
            for j in range((self.size**2)):
                if self.adjacency[i,j] == 1:
                    self.g.addEdge(i,j, self.weight_matrix[i,j])

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
                    x_vals = [i % self.size, j% self.size] 
                    y_vals = [i // self.size, j // self.size]
                    if self.weight_matrix[i,j] == 1:
                        ax.plot(x_vals, y_vals, 'o', linestyle = '-', color = 'black', alpha =0.5, zorder = 10)
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
            fig.suptitle(f"Path: {','.join([str(x) for x in path])}")
        elif plot_path and not path:
            fig.suptitle("No solution found", y = 0.93)

        plt.show()



if __name__ == '__main__':
    r = RoadNetwork(10)
    r.generate_adjacency()
    r.generate_weights(charging=False)
    r.plot_network()
    r.load_graph(10)
    r.get_path(0, 99)
    r.generate_weights(charging=True)
    r.plot_network()
    r.load_graph(10)
    r.get_path(0, 99)
