import numpy as np

class Link():
    """
    A link object. The link has three nodes: 0, 1, 2. Visually, arranged as such:
    2----------------
    |               |
    0 ------------- 1
    The primary link is between nodes 0 and 1. This link is uni-directional.
    As a default, each link has one weight. This weight is the cost of traversing the link.
    """

    def __init__(self, num_weights = 1):
        self.adjacency = np.zeros((3,3))

        # Initialize weights
        individual_weight = [0]*num_weights
        self.weights = np.array([[individual_weight]*3]*3)

        # Set the default link
        self.adjacency[0,1] = 1

    def set_main_weight(self, weight):
        """
        Set the weight of the primary link.
        """
        self.weights[0,1] = weight

    def unlink(self):
        """
        Unlink the primary nodes.
        """
        self.adjacency[0,1] = 0
        

    def set_link(self, node1, node2, weight):
        """
        Set a link between two nodes.
        """
        self.adjacency[node1, node2] = 1
        self.weights[node1, node2] = weight



    
