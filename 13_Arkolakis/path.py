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

    def weights_leq(self, constraints):
        """
        Returns whether or not the path is within some set of constraints.
        Note, assumes the number of constraints is the same as the number of 
        edge weights on each link in the path. (i.e. the number of QoS metrics)
        """
        return np.all(np.less_equal(self.weight_sum, constraints))

    def repeat_visits(self):
        counts = [self.path.count(x) for x in list(set(self.path))]
        if np.any([x > 1 for x in counts]):
            return True
        else:
            return False
    
    def __add__(self, p2):
        if (p2.path[0] != self.path[-1]):
            raise ValueError("You cannot do this")
        
        p3 = Path(self.path[-1])
        p3.path = self.path + p2.path[1:]
        p3.weights = self.weights + p2.weights
        p3.updateWeightSum()

        return p3