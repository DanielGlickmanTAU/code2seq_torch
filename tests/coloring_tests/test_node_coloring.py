from coloring.datasets import PyramidNodeColorDataset
import visualization
from coloring import coloring_utils
dataset = PyramidNodeColorDataset(max_row_size=5, num_adj_stack=5)

index_to_color = coloring_utils.index_to_color
visualization.draw(dataset.dataset[0], dataset.dataset[0].y, color_map=index_to_color)
visualization.draw(dataset.dataset[0], dataset.dataset[0].x, color_map=coloring_utils.index_to_color_map_with_white(index_to_color))
print(dataset)