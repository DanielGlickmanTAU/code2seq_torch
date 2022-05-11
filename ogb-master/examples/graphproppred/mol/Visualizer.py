import visualization
from exp_utils import get_global_exp


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
        if not self.task_type == 'coloring':
            return

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
                    visualization.draw_pyramid(g, 'x', 'input',fig_name=fig_name)
                    visualization.draw_pyramid(g, 'y', 'gold',fig_name=fig_name)
                label = f'epoch {self._lexo(self.epoch, 1000)}. acc:{g_acc}'
                visualization.draw_pyramid(g, g_pred, label,fig_name=fig_name)
            except Exception as e:
                print(f'failed visualizing {e}')

        get_global_exp().log_confusion_matrix(y_true.numpy(), y_predicted=y_pred.numpy(), labels=self.label_names)
