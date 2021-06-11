from pyvis.network import Network

from src.data.graph import Graph


def draw_graph(graph: Graph, height: int = 1000, width: int = 1000, notebook: bool = True) -> Network:
    """Visualize graph using [pyvis](https://pyvis.readthedocs.io/en/latest/) library

    :param graph: graph instance to visualize
    :param height: height of target visualization
    :param width: width of target visualization
    :param notebook: pass True if visualization should be displayed in notebook
    :return: pyvis Network instance
    """
    net = Network(height=height, width=width, directed=True, notebook=True)
    net.barnes_hut(gravity=-10000, overlap=1, spring_length=1)

    for node in graph.nodes:
        net.add_node(node.id, label=node.token, group=node.type.value)

    for edge in graph.edges:
        net.add_edge(edge.from_node.id, edge.to_node.id, label=edge.type.name)

    return net
