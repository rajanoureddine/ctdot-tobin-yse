from path import Path
import numpy as np

class GraphExtended():
    """
    An implementation of the Extended Bellman-Ford algorithm, as described in 
    - Meongchul Song and Sahni, S. (2006) ‘Approximation algorithms for multiconstrained quality-of-service routing’,
        IEEE Transactions on Computers, 55(5), pp. 603–617. Available at: https://doi.org/10.1109/TC.2006.67.
    - Garroppo, R.G., Giordano, S. and Tavanti, L. (2010) ‘A survey on multi-constrained optimal path computation: 
        Exact and approximate algorithms’, Computer Networks, 54(17), pp. 3081–3107. Available at: https://doi.org/10.1016/j.comnet.2010.05.017.
    """

    def __init__(self, vertices, weights, constraints):
        # Weights is redundant - remove
        self.V = vertices   # No. of vertices
        self.graph = []     # Empty graph
        self.path = [[]] * self.V 
        self.constraints = constraints

    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])
    
    def BellmanExtended(self, src, dest, print_result = True):
        # Generate an empty path. The path to source will be the source node itself, weights = 0
        self.path[src] = [Path(src)]

        # Repeat n-1 times
        for m in range(self.V - 1):
            for u, v, w in self.graph:
                for p in self.path[u]:
                    uv = Path(u) # Create a new path that starts at u
                    uv.addLink(u, v, w) # Add a link to v, with relevant weights

                    # New to add
                    new_path = p + uv # Add this new route to v

                    if new_path.weights_leq(self.constraints) and not new_path.repeat_visits():

                        # If it's within the constraints, check 
                        flag = True
                        add_path_flag = True

                        # If there are any existing paths to v, iterate through them. We either add the new path (keeping q),
                        # add it dropping q, or don't add it and keep q. 
                        if self.path[v]:
                            for q in self.path[v]:
                                # Case 1: If q is better on every metric than new path
                                if np.all(np.less(q.weight_sum, new_path.weight_sum)):
                                    flag = False
                                    break # don't need to consider any more 

                                # Case 2: If np is either the same as q for all metrics, less than
                                # q for all metrics, or a mix of being the same and better
                                if np.all(np.less_equal(new_path.weight_sum, q.weight_sum)):
                                    add_path_flag = True # We will be adding a path
                                    
                                    # Case 2b: If new path is not the same for all metrics (that means
                                    # it's the same for some metrics, and better for others).
                                    # Thus we remove q
                                    if not np.all(np.equal(new_path.weight_sum, q.weight_sum)):
                                        self.path[v].remove(q)
                                    
                                    # If they're the same path, don't add a new path
                                    elif new_path.path == q.path:
                                        add_path_flag = False
                                        break
                                        
                        if flag == True:
                            if self.path[v] and add_path_flag:
                                self.path[v].append(new_path)
                            elif not self.path[v]:
                                self.path[v] = [new_path]

        if print_result:
            self.printPaths()
        #else:
        #    print(self.path[dest][0].path)

    def printPaths(self, dest = False):
        print("Vertex Distance from Source, and Path")
        print("-------------------------------------")
        for i in range(self.V):
            if len(self.path[i]) >1:
                paths = [p.path for p in self.path[i]]
                weights = [p.weight_sum for p in self.path[i]]
            else:
                try:
                    paths = [self.path[i][0].path]
                    weights = [self.path[i][0].weight_sum]
                except:
                    pass
            for j in range(len(paths)):
                print(f"Node {i} path {j}\tWeights: {weights[j]}\tPath:{paths[j]}")
            print("-------------------------------------")

if __name__ == '__main__':
    g = GraphExtended(8, 2, np.array([np.infty, 47]))
    for i in range(1, 4):
        g.addEdge(i-1, i, [10, 17])
    
    g.addEdge(1, 4, [20,-30])
    g.addEdge(4,2, [10,17])
    g.addEdge(0,5,[12,16])
    g.addEdge(5,6, [12,16])
    g.addEdge(6,3, [12,16])
    g.addEdge(0,7, [10, 20])
    g.addEdge(7,3,[10,20])
    g.BellmanExtended(0,3)