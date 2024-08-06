"""
Implement vanilla versions of Dijkstra's algorithm and the Bellman-Ford algorithm.
"""

import numpy as np

class Graph:
    def __init__(self, vertices, weights = 1):
        self.weights = weights
        self.V = vertices  # No. of vertices
        self.graph = []
        self.dist = []

        # Define path or previous 
        if self.weights > 1:
            self.path = [{"path": None, "weights": []}] * self.V
            self.path_weights = [None] * self.V
        elif self.weights == 1:
            self.prev = [None] * self.V
 
    # function to add an edge to graph
    def addEdge(self, u, v, w):
        if (not type(w) == list) and self.weights > 1:
            raise ValueError(f"You must provide {self.weights} weights")
        elif type(w) == list and len(w) != self.weights:
            raise ValueError(f"You must provide {self.weights} weights")
        else:
            self.graph.append([u, v, w])

    
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
    def printArr(self, src):
        print("Vertex Distance from Source, and Path")
        for i in range(self.V):
            path = self.getPath(i, src)
            print("{0}\t{1}\t\t{2}".format(i, self.dist[i], path))
    
    # utility function used to print the path
    def printPaths(self, src):
        for i in range(self.V):
            print(f"Path to {i}: {self.getPath(i, src)}")
 
    # The main function that finds shortest distances 
    def BellmanFord(self, src, print = False):
        if not self.weights == 1:
            raise ValueError("Standard Bellman-Ford Algorithm only works with one weight")
 
        # Step 1: Initialize distances from src to all other vertices
        self.dist = [float("Inf")] * self.V
        self.dist[src] = 0
 
        # Step 2: Relax all edges |V| - 1 times. A simple shortest
        for _ in range(self.V - 1):

            for u, v, w in self.graph:
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
        for u, v, w in self.graph:
            if self.dist[u] != float("Inf") and self.dist[u] + w < self.dist[v]:
                print("Graph contains negative weight cycle")
                return
        
        if print:
            self.printArr(src)
 
    def Dijkstra(self, src):
        self.dist = [float("Inf")] * self.V
        self.dist[src] = 0

        A = []
        B = list(np.arange(self.V))

        while len(B) > 0:
            # Node in B with minimum distance
            distances_b = [self.dist[i] for i in B]
            u = B[np.argmin(distances_b)]

            # B = B \ {u}
            B.remove(u)

            # A = A + u
            A.append(u)

            for u2, v, w in self.graph:
                if ((u2 == u) and v in B): # Check the neighbours of u only
                    dsuv = self.dist[u] + w
                    if dsuv < self.dist[v]:
                        self.dist[v] = dsuv
                        self.prev[v] = u
                    elif dsuv == self.dist[v]:
                        if self.prev[v] is None:
                            self.prev[v] = u
                        elif type(self.prev[v]) != list:
                            self.prev[v] = sorted([self.prev[v], u])
                        else:
                            self.prev[v] = sorted(self.prev[v]+ [u])

        # self.printArr(src)
    
    def reset(self):
        self.dist = []

        # Define path or previous 
        if self.weights > 1:
            self.path = [None] * self.V
            self.path_weights = [None] * self.V
        
        elif self.weights == 1:
            self.prev = [None] * self.V
