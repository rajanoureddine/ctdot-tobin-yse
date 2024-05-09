import numpy as np

class Link():
    """
    A link object. The link has three nodes: 0, 1, 2. Visually, arranged as such:
    2----------------
    |               |
    0 ------------- 1
    As a default, there is a one-way link between nodes 0 and 1. This represents a standard link.
    As a default, each link has one weight. This weight is the cost of traversing the link.
    """

    def __init__(self, num_weights = 1):
        self.adjacency = np.zeros((3,3))

        # Initialize weights
        individual_weight = [0]*num_weights
        self.weights = np.array([[individual_weight]*3]*3)

        # Set the default link
        self.adjacency[0,1] = 1

    def set_link(self, node1, node2, weight):
        """
        Set a link between two nodes.
        """
        self.adjacency[node1, node2] = 1
        self.weights[node1, node2] = weight

if __name__ == "__main__":
    link = Link()
    print(link.adjacency)
    print(link.weights)
    print(link.adjacency[0,1])
    print(link.weights[0,1])
    
