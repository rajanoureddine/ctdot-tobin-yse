import numpy as np
import pytest
from bellman_ford import Graph, Path, GraphExtended


@pytest.fixture
def setup_graph():
    graph_size = 20
    graph = np.zeros([graph_size, graph_size])
    for i in range(graph_size):
        for j in range(graph_size):
            graph[i][j] = np.random.choice([0, 1], p=[0.7, 0.3])
    graph_weights = np.random.randint(1, 5, (graph_size, graph_size))
    g = Graph(graph_size)
    for i in range(graph_size):
        for j in range(graph_size):
            if graph[i][j] != 0:
                g.addEdge(i, j, graph_weights[i][j])
    yield(g, graph, graph_weights)


@pytest.fixture
def setup_bf_extended():
    g = GraphExtended(14, 2, np.array([np.infty, 40]))
    for i in range(1, 6):
        g.addEdge(i-1, i, [6, 10])
    
    g.addEdge(0, 6, [9,15])
    g.addEdge(6, 7, [6,10])
    g.addEdge(7,8, [19, -30])
    g.addEdge(8,9, [6,10])
    g.addEdge(9,5, [9,15])
    g.addEdge(0,10, [9,15])
    g.addEdge(10,11, [6,10])
    g.addEdge(11,12, [19,-31])
    g.addEdge(12,13, [6,10])
    g.addEdge(13, 5, [9,15])

    g.BellmanExtended(0, 5)
    yield(g)

def test_weight_numbers():
    # Test that it won't accept too many weights
    g = Graph(5, 1)
    with pytest.raises(ValueError):
        g.addEdge(0, 1, [-1, -3])
    
    # Test that it won't accept too few weights
    g = Graph(5, 3)
    with pytest.raises(ValueError):
        g.addEdge(0, 1, -1)
    
    # Test that it will accept the right number of weights
    g.addEdge(0, 1, [-1, -3, -5])

def test_bellman_ford_add_constraint():
    g = Graph(5, 1)
    g.addEdge(0, 1, -1)
    g.addConstraint(1, 5)

    with pytest.raises(ValueError):
        g.addConstraint(3, 5)
    
def test_BF_too_many_weights():
    g = Graph(5, 4)
    g.addEdge(0, 1, [-1, -3, -5, -7])
    with pytest.raises(ValueError):
        g.BellmanFord(0)

def test_bellman_ford(setup_graph):
    g, _, _ = setup_graph
    g.BellmanFord(0)

def test_dijkstra(setup_graph):
    g, _, _ = setup_graph
    g.Dijkstra(0)

def test_bellman_ford_dijkstra(setup_graph, capfd):
    g, graph, graph_weights = setup_graph
    g.BellmanFord(0)
    out1, err1 = capfd.readouterr()
    g.reset()
    g.Dijkstra(0)
    out2, err2 = capfd.readouterr()
    print(graph)
    print(graph_weights)

    assert(out1 == out2)

def test_Path():
    p1 = Path(0)
    p1.addLink(0,1,[1])
    assert(p1.path == [0,1])
    assert(p1.weights == [[1]])
    assert(p1.weight_sum == [1])

    with pytest.raises(ValueError):
        p1.addLink(3, 4, [0,1])

def test_Path_add():
    p1 = Path(0)
    p1.addLink(0,1,[1])

    p2 = Path(1)
    p2.addLink(1, 2, [3])

    p3 = p1 + p2
    
    assert(p3.path == [0,1,2])
    assert(p3.weights == [[1],[3]])
    assert(p3.weight_sum == [4])

    p1 = Path(0)
    p1.addLink(0,1,[1,2])

    p2 = Path(1)
    p2.addLink(1, 2, [3,4])

    p3 = p1 + p2
    
    assert(p3.path == [0,1,2])
    assert(p3.weights == [[1,2],[3,4]])
    assert(np.array_equal(p3.weight_sum, np.array([4,6])))

def test_BF_extended(setup_bf_extended, capfd):
    g = setup_bf_extended
    g.BellmanExtended(0, 5)
    out, err = capfd.readouterr()
    assert(out == "[0, 10, 11, 12, 13, 5]\n")




# def test_bellman_ford(capfd):
#     g = Graph(graph_size)
#     for i in range(graph_size):
#         for j in range(graph_size):
#             if graph[i][j] != 0:
#                 g.addEdge(i, j, graph_weights[i][j])
#     g.BellmanFord(0)
#     out1, err1 = capfd.readouterr()
#     g.reset()
#     g.Dijkstra(0)
#     out2, err2 = capfd.readouterr()
#     assert(out1 == out2)
#     # print(graph)
#     # print(graph_weights)

#def test_bellman_ford(benchmark):
#    result = benchmark(bellman_ford)
#    assert result