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
        self.V = vertices   # No. of vertices
        self.graph = []     # Empty graph
        self.path = [[]] * self.V 
        self.constraints = constraints

    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])
    
    def BellmanExtended(self, src, dest, print_result = True):
        # Generate an empty path. The path to source will be the source node itself, weights = 0
        self.path[src] = [Path(src)]

        # Repeat n-1 times, where n is the number of vertices
        for m in range(self.V - 1):
            # For each u,v edge in the graph, and the associated edge weights w 
            # (w could be a list of weights if there are multiple metrics!)
            for u, v, w in self.graph:
                # For each already-existing path to u, p
                for p in self.path[u]:
                    uv = Path(u) # Create a new path that starts at u
                    uv.addLink(u, v, w) # Add a link from u to v, with relevant weights

                    # Create a new path to p, that is the old path p to u, plus the extra link
                    new_path = p + uv 

                    # If the path does not exceed the constraints, and does not include repeat visits
                    if new_path.weights_leq(self.constraints) and not new_path.repeat_visits():

                        # If it's within the constraints, check 
                        flag = True
                        add_path_flag = True

                        # If there are any existing paths to v... 
                        if self.path[v]:
                            # Iterate through these existing paths to see if new_path is better
                            for q in self.path[v]:
                                # Case 0: If they are the same path, don't add a new path
                                if new_path.path == q.path:
                                    add_path_flag = False
                                    break

                                # Case 1: if q is less than new_path for the minimization metric, then we don't need to consider the new path
                                elif (len(self.constraints) == 1) and np.less(q.weight_sum, new_path.weight_sum):
                                    flag = False
                                    
                                elif (len(self.constraints) > 1) and np.less(q.weight_sum[0], new_path.weight_sum[0]):
                                    flag = False

                                # Case 2: If q is greater than new_path for the minimization metric, then we drop q
                                elif (len(self.constraints) == 1) and np.greater(q.weight_sum, new_path.weight_sum):
                                    add_path_flag = True # We will be adding a path
                                    self.path[v].remove(q)
                                elif (len(self.constraints) > 1) and np.greater(q.weight_sum[0], new_path.weight_sum[0]):
                                    add_path_flag = True # We will be adding a path
                                    self.path[v].remove(q)

                                # Case 3a: If q is equal for all metrics, then we add the new path but do not drop q
                                elif np.all(np.equal(q.weight_sum, new_path.weight_sum)):
                                    add_path_flag = True
                                    # break
                                
                                # Note: at this point we're left only with cases where there are more than one metric
                                # Case 3b: If q is equal for the optimization metric, and q is better for all others
                                elif np.all(np.less(q.weight_sum[0:], new_path.weight_sum[0:])):
                                    Flag = False # We will not add the new path
                                    # break

                                # Case 3c: If q is the same as new_path for the minimization metric, and q is worse for all others
                                elif np.all(np.greater(q.weight_sum[0:], new_path.weight_sum[0:])):
                                    add_path_flag = True
                                    self.path[v].remove(q)
                                    # break
                                # Case 3c: If q is the same as new_path for the minimization metric, and q is better for some and worse for others
                                else:
                                    add_path_flag = True # We will be adding a path (we keep both paths)

                                        
                        if flag == True:
                            if self.path[v] and add_path_flag and not len(self.path[v]) > 1:
                                self.path[v].append(new_path)
                            elif not self.path[v]:
                                self.path[v] = [new_path]
                            if v == dest:
                                break

        if print_result:
            self.printPaths()
        #else:
        #    print(self.path[dest][0].path)

    def printPaths(self, dest = False):
        print("Vertex Distance from Source, and Path")
        print("-------------------------------------")
        for i in range(self.V):
            paths = []
            weights = []
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
            if paths:
                print("-------------------------------------")


# A quick test of the implementation.
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