import numpy as np

class Path:
    def __init__(self, src):
        self.path = [src]
        self.weights = []
        self.weight_sum = 0
    
    def addLink(self, u, v, w):
        if self.path == [u]:
            self.path = [u, v]
            self.weights = [w]
        elif self.path[-1] != u:
            raise ValueError(f"You cannot add a link from {u} to {v}, since the last link is {self.path[-1]}")
        else:
            self.path.append(v)
            self.weights.append(w)
        
        # Either way
        self.updateWeightSum()
    
    def updateWeightSum(self):
        self.weight_sum = np.sum(self.weights, axis = 0)
    
    def __add__(self, p2):
        if (p2.path[0] != self.path[-1]):
            raise ValueError("You cannot do this")
        
        p3 = Path(self.path[-1])
        p3.path = self.path + p2.path[1:]
        p3.weights = self.weights + p2.weights
        p3.updateWeightSum()

        return p3


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

        # Initialize constraints
        self.constraints = {}
        for i in range(1, self.weights):
            self.constraints[i] = []
 
    # function to add an edge to graph
    def addEdge(self, u, v, w):
        if (not type(w) == list) and self.weights > 1:
            raise ValueError(f"You must provide {self.weights} weights")
        elif type(w) == list and len(w) != self.weights:
            raise ValueError(f"You must provide {self.weights} weights")
        else:
            self.graph.append([u, v, w])
    
    def addConstraint(self, weight_index, max):
        if weight_index > self.weights:
            raise ValueError("There are only {self.weights} weights")
        else:
            self.constraints[weight_index] = max
        for key, value in self.constraints.items():
            print(f"Constraint {key}: {value}")

    
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


class GraphExtended():
    def __init__(self, vertices, weights, constraints):
        self.weights = weights
        self.V = vertices  # No. of vertices
        self.graph = []
        self.path = [[]] * self.V
        self.constraints = constraints

    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])
    
    def BellmanExtended(self, src, dest, print_dest = False):
        self.path[src] = [Path(src)]

        for _ in range(self.V - 1):

            for u, v, w in self.graph:
                for p in self.path[u]:
                    # Create paths
                    uv = Path(u)
                    uv.addLink(u, v, w)

                    # New to add
                    new_path = p + uv

                    # Check it's within the constraints
                    if np.all(np.less_equal(new_path.weight_sum, self.constraints)):

                        # If it's within the constraints, check 
                        flag = True
                        add_path_flag = True

                        if self.path[v]:
                            for q in self.path[v]:
                                if np.all(np.less(q.weight_sum, new_path.weight_sum)):
                                    flag = False
                                    break # don't need to consider any more 
                                if np.all(np.less_equal(new_path.weight_sum, q.weight_sum)):
                                    add_path_flag = True
                                    if not np.all(np.equal(new_path.weight_sum, q.weight_sum)):
                                        self.path[v].remove(q)
                                    elif new_path.path == q.path:
                                        add_path_flag = False
                                        break
                                        
                        if flag == True:
                            if self.path[v] and add_path_flag:
                                self.path[v].append(new_path)
                            else:
                                self.path[v] = [new_path]

        if not print_dest:
            self.printPaths()
        else:
            print(self.path[dest][0].path)

    def printPaths(self, dest = False):
        print("Vertex Distance from Source, and Path")
        for i in range(self.V):
            if len(self.path[i]) >1:
                paths = [p.path for p in self.path[i]]
            else:
                paths = self.path[i][0].path
            print("{0}\t{1}\t\t{2}".format(i, self.path[i][0].weight_sum, paths))


# if __name__ == '__main__':
#    g = GraphExtended(14, 2, np.array([np.infty, 40]))
#    for i in range(1, 6):
#        g.addEdge(i-1, i, [6, 10])
#    
#    g.addEdge(0, 6, [9,15])
#    g.addEdge(6, 7, [6,10])
#    g.addEdge(7,8, [19, -30])
#    g.addEdge(8,9, [6,10])
#    g.addEdge(9,5, [9,15])
#    g.addEdge(0,10, [9,15])
#    g.addEdge(10,11, [6,10])
#    g.addEdge(11,12, [19,-31])
#    g.addEdge(12,13, [6,10])
#    g.addEdge(13, 5, [9,15])
# 
#    g.BellmanExtended(0, 5)

   #for i in g.path:
   #     print(i[0].path)

if __name__ == '__main__':
    graph_size = 20
    graph = np.zeros([graph_size, graph_size])
    for i in range(graph_size):
        for j in range(graph_size):
            graph[i][j] = np.random.choice([0, 1], p=[0.6, 0.4])
    graph_weights = np.random.randint(1, 8, (graph_size, graph_size))
    g = Graph(graph_size)
    g_e = GraphExtended(graph_size, 1, np.array([np.infty]))
    for i in range(graph_size):
        for j in range(graph_size):
            if graph[i][j] != 0:
                g.addEdge(i, j, graph_weights[i][j])
                g_e.addEdge(i, j, graph_weights[i][j])
    #g.Dijkstra(0)
    #print(g.prev)
    #g.reset()
    g.BellmanFord(0, True)
    g_e.BellmanExtended(0, 5)
    # print(g_e.path[5][0].path)
   #  print(graph)