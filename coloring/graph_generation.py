import networkx as nx


def create_pyramid(min_row_size, max_row_size):
    graph = nx.Graph()
    # +1 here just makes it max row size correct(and not -1)
    for row in range(min_row_size, max_row_size + 1):
        row_index = row - min_row_size
        for col in range(1, row):
            graph.add_edge((row_index, col), (row_index, col - 1))
    for row in range(min_row_size, max_row_size):
        row_index = row - min_row_size
        for col in range(row):
            graph.add_edge((row_index, col), (row_index + 1, col))
            graph.add_edge((row_index, col), (row_index + 1, col + 1))

    positions = {(row, col): (col + 0.5 * abs(row - max_row_size), -row) for (row, col) in graph.nodes}
    return graph, positions


def create_hexagon_from_triangles(max_row_size, min_row_size):
    graph, positions = create_pyramid(min_row_size, max_row_size)
    # lower graph part:
    for row in range(max_row_size + 1, 2 * max_row_size - min_row_size + 1):
        row_size = 2 * max_row_size - row
        for col in range(1, row_size):
            graph.add_edge((row, col), (row, col - 1))
        for col in range(row_size):
            graph.add_edge((row, col), (row - 1, col))
            graph.add_edge((row, col), (row - 1, col + 1))
    #todo fix positions offset to start at 0
    return graph, positions