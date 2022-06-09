import collections

import visualization
from exp_utils import get_global_exp
import torch
from ogb.graphproppred.evaluate import accuracy_coloring


class Visualizer:
    def __init__(self, task_name, epoch, graph_indexes=[0, 1], label_names=None):
        self.label_names = label_names
        self.epoch = epoch
        self.task_type = task_name
        self.graph_indexes = graph_indexes

    @staticmethod
    def _lexo(n, max_val):
        s = str(n)
        n = n * 10
        while n <= max_val:
            s = '0' + s
            n = n * 10
        return s

    def __call__(self, graphs, y_true, y_pred):
        # if not self.task_type == 'coloring':
        #     return

        graph_to_prediction_indexes = {}
        predictions_start = 0
        for i, graph in enumerate(graphs):
            predictions_end = predictions_start + graph.num_nodes
            graph_to_prediction_indexes[i] = (predictions_start, predictions_end)
            predictions_start = predictions_end

        for index in self.graph_indexes:
            try:
                g = graphs[index]

                pred_start, pred_end = graph_to_prediction_indexes[index]
                g_pred = y_pred[pred_start:pred_end]
                g_acc = (g_pred.argmax(dim=-1) == g.y).float().mean()
                g_acc = str(round(g_acc.item(), 2))
                fig_name = f'graph {index}'
                if self.epoch == 1:
                    visualization.draw_pyramid(g, 'x', 'input', fig_name=f'{fig_name}_input')
                    visualization.draw_pyramid(g, 'y', 'gold', fig_name=f'{fig_name}_gold')

                    visualization.draw_pyramid(g, 'x', 'input', fig_name=fig_name)
                    visualization.draw_pyramid(g, 'y', 'gold', fig_name=fig_name)
                label = f'epoch {self._lexo(self.epoch, 1000)}. acc:{g_acc}'
                visualization.draw_pyramid(g, g_pred, label, fig_name=fig_name)
            except Exception as e:
                print(f'failed visualizing {e}')
        if 'shape' in graphs[0]:
            name_to_node_indexes = collections.defaultdict(list)
            shape_to_acc = {}
            i = 0
            for graph in graphs:
                for node_shape in graph['shape']:
                    name_to_node_indexes[node_shape].append(i)
                    i = i + 1
            assert sum(len(v) for v in name_to_node_indexes.values()) == len(y_true)
            for shape_name in name_to_node_indexes:
                indexes_of_nodes_in_shape = torch.tensor(name_to_node_indexes[shape_name])
                shape_acc = accuracy_coloring(y_true[indexes_of_nodes_in_shape].numpy(),
                                              y_pred[indexes_of_nodes_in_shape].numpy())
                shape_to_acc[shape_name] = shape_acc
            get_global_exp().log_metrics(shape_to_acc, prefix='shape')
        else:
            get_global_exp().log_confusion_matrix(y_true.numpy(), y_predicted=y_pred.numpy(), labels=self.label_names)
