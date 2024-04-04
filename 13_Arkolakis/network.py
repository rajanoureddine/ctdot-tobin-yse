import numpy as np
from matplotlib import pyplot as plt
from bf_extended import GraphExtended

class RoadNetwork():
    def __init__(self, dim):
        self.size = dim            # Number of rows and columns. Num nodes = self.size ** 2
        self.adjacency = np.zeros((self.size**2, self.size**2))
        self.g = GraphExtended(self.size**2, 1, np.array([np.infty]))
        self.generate_adjacency()
        self.load_graph()

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
                    self.adjacency[i, j] = np.random.choice([0,1], 1, p=[0.6, 0.4])
    
    
    def load_graph(self, weight = 1):
        """Loads the graph with the adjacency data.
        For now, assume each link has length 1"""

        for i in range((self.size**2)):
            for j in range((self.size**2)):
                if self.adjacency[i,j] == 1:
                    self.g.addEdge(i,j,weight)

    def get_path(self, src, dest):
        self.g.BellmanExtended(src, dest, False)

        path = self.g.path[dest]

        if len(path)>0:
            solution = path[0].path
            self.plot_network(solution)
        


    
    def plot_network(self, path = None):
        """Given a valid adjacency matrix, plot the network"""

        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,10), facecolor = 'whitesmoke')

        ax.set_xlim((-1, self.size))
        ax.set_ylim((-1, self.size))

        # Iterate through each of the points and plot
        for i in range(self.size**2):
            for j in range(self.size**2):
                if self.adjacency[i,j] == 1:  # There is a connection between node i, and node j
                    # print(f"{i} to {j}")
                    x_vals = [i % self.size, j% self.size] 
                    y_vals = [i // self.size, j // self.size]
                    ax.plot(x_vals, y_vals, 'bo', linestyle = '-', alpha =0.5)
        
        if path:
            for z in range(1, len(path)):
                i, j = path[z-1], path[z]
                x_vals = [i % self.size, j% self.size] 
                y_vals = [i // self.size, j // self.size]
                ax.plot(x_vals, y_vals, 'bo', linestyle = '-', color = 'red')

        plt.show()



if __name__ == '__main__':
    r = RoadNetwork(10)
    r.plot_network()
