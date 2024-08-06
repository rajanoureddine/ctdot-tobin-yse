"""
Implements Dijkstra's algorithm for shortest path in a graph.
Creates a Graph class, with a number of vertices and weights.
"""
import numpy as np

class Graph:
    def __init__(self, vertices, constraint = None, objective = "nonlinear"):
        """
        Initializes a graph of the form [u, v, w, b] where u is the source, v is the destination, and w is the weight.
        b is the boost value that slackens the constraints.
        The graph is represented as a list of lists.

        Parameters:
        vertices (int): Number of vertices in the graph
        constraint(int): Constraint on the paths
        """ 
        self.V = vertices  # No. of vertices
        self.constraint = constraint
        self.graph = []
        self.total_cost = []
        self.prev = []
        self.objective_type = objective
        self.alpha1 = 0.5
        self.alpha2 = 0.5
 
    # function to add an edge to graph
    def addEdge(self, u, v, w, b = None):
        if b:
            self.graph.append([u, v, w, b])
        else:
            self.graph.append([u, v, w, 0])
    
    # Function to set parameters
    def set_params(self, alpha1, alpha2):
        self.alpha1 = alpha1
        self.alpha2 = alpha2
    
    # Define objective function
    def objective(self, cost, boost):
        # Get the parameters
        alpha1 = self.alpha1
        alpha2 = self.alpha2

        # Linear objective
        if self.objective_type == "linear":
            return (alpha1 * cost) - alpha2*(self.constraint - cost + boost)
        
        # Nonlinear objective
        elif self.objective_type == "nonlinear":
            return (alpha1 * cost) - alpha2*np.sqrt((self.constraint - cost + boost))
    
 
    def Dijkstra(self, src, dest, print_it = True, return_path = False):
        output = False

        # Initialize
        self.total_cost = [float("Inf")] * self.V
        self.boost = [0] * self.V
        self.cost = [0] * self.V
        self.prev = [None] * self.V

        # Initialize the distance of the source node to 0
        self.total_cost[src] = 0

        A = []
        B = list(np.arange(self.V))

        while len(B) > 0:
            # Node in B with minimum distance
            distances_b = [self.total_cost[i] for i in B]
            u = B[np.argmin(distances_b)]

            # B = B \ {u}
            B.remove(u)

            # A = A + u
            A.append(u)

            for u2, v, w, b in self.graph:
                if ((u2 == u) and v in B): # Check the neighbours of u only
                    alt_cost = self.cost[u] + w
                    alt_boost = self.boost[u] + b
                    alt = self.objective(alt_cost, alt_boost)
                    # print(f"Considering {u} -> {v} with weight {w} and boost {b} with cost {alt_cost} and boost {alt_boost} and objective {alt}")
                    if v == dest:
                        output = True
                    if alt < self.total_cost[v]:
                        self.total_cost[v] = alt
                        self.cost[v] = alt_cost
                        self.boost[v] = alt_boost
                        self.prev[v] = u


        if print_it:
            self.printArr(src)

        if return_path:
            return self.getPath(dest, src), self.total_cost[dest], self.cost[dest], self.boost[dest]

        return output
    
    def BellmanFord(self, src, dest, print_it = True, return_path = False):
        # Step 1: Initialize distances from src to all other vertices
        self.total_cost = [float("Inf")] * self.V
        self.boost = [0] * self.V
        self.cost = [0] * self.V
        self.prev = [None] * self.V
 
        # Step 2: Relax all edges |V| - 1 times. A simple shortest
        for _ in range(self.V - 1):

            for u, v, w, b in self.graph:
                alt_cost = self.cost[u] + w
                alt_boost = self.boost[u] + b 
                alt = self.objective(alt_cost, alt_boost)

                if v == dest:
                    output = True

                if alt < self.total_cost[v]:
                    self.total_cost[v] = alt
                    self.cost[v] = alt_cost
                    self.boost[v] = alt_boost
                    self.prev[v] = u

                if False:
                    if self.dist[u] != float("Inf") and self.dist[u] + w < self.dist[v]:
                        self.dist[v] = self.dist[u] + w
                        self.prev[v] = u
                    
                    # Allows for possibility of equally distant paths
                    elif self.dist[u] != float("Inf") and self.dist[u] + w == self.dist[v]:
                        if self.prev[v] is None:
                            self.prev[v] = u
                        elif type(self.prev[v]) != list and self.prev[v] != u:
                            self.prev[v] = sorted([self.prev[v], u])
                        elif type(self.prev[v]) == list and u not in self.prev[v]:
                            self.prev[v] = sorted(self.prev[v] + [u])
                        
        # Step 3: check for negative-weight cycles.
        if False:
            for u, v, w in self.graph:
                if self.dist[u] != float("Inf") and self.dist[u] + w < self.dist[v]:
                    print("Graph contains negative weight cycle")
                    return
        
        if print_it:
            self.printArr(src)

        if return_path:
            return self.getPath(dest, src), self.total_cost[dest], self.cost[dest], self.boost[dest]

        return output


    # Function to get the path to a node
    def getPath(self, node, src, path = None):
        if not path:
            path = [node]
        if node == src:
            return path[::-1] # reverse it 
        else:
            try:
                if type(self.prev[node]) != list:
                    path.append(self.prev[node])
                    return self.getPath(self.prev[node], src, path)
                else:
                    return [self.getPath(self.prev[node][i], src, path + [self.prev[node][i]]) for i in range(len(self.prev[node]))]

            except Exception as e:
                return "No path"

    # utility function used to print the solution
    def printArr(self, src, dest=None):
        print("Vertex Distance from Source, and Path")
        if dest:
            path = self.getPath(dest, src)
            print("{0}\t{1}\t\t{2}".format(dest, self.total_cost[dest], path))

        if not dest:
            for i in range(self.V):
                if self.total_cost[i] != float("Inf"):
                    path = self.getPath(i, src)
                    print("{0}\t{1:.3f}\t\t{2}\t{3}\t{4}".format(i, self.total_cost[i], self.cost[i], self.boost[i], path))
    
    # utility function used to print the path
    def printPaths(self, src):
        for i in range(self.V):
            print(f"Path to {i}: {self.getPath(i, src)}")


# If name is main
if __name__ == "__main__":
    # Test the Dijkstra algorithm
    g = Graph(6, 3)

    # Direct path
    g.addEdge(0,1,1)
    g.addEdge(1,2,1)
    g.addEdge(2,3,1)

    # Now add charging/loops
    g.addEdge(1,4,0.5)
    g.addEdge(4,2,1,1)
    g.addEdge(2,5, .5)
    g.addEdge(5,3,1,1)

    # Run
    g.Dijkstra(0, 3)