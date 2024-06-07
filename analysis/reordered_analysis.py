import pandas as pd
from matplotlib import pyplot as plt

from analysis import Analysis


class ReorderedAnalysis(Analysis):

    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.tip_probabilities = None
        self. reordered_probabilities = None
        self.order_counts_per_product = None

    def _analyze(self):
        self.tip_probabilities = self.orders_joined.groupby('product_id')['tip'].mean()
        self.reorder_probabilities = self.orders_joined.groupby('product_id')['reordered'].mean()

    def _show_results(self, save_plots=False):
        self._plot_scatter(self.tip_probabilities, self.reorder_probabilities, save_plots)

    def _plot_scatter(self, tip_probabilities, reorder_probabilities, save_plots=False):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(reorder_probabilities, tip_probabilities, alpha=0.6, edgecolors='w')
        ax.set_xlabel('Reorder Probability')
        ax.set_ylabel('Tip Probability')
        ax.set_title('Tip Probability vs Reorder Probability by Product')

        plt.show()

        if save_plots:
            self._save_plot(fig, 'reordered_analysis.png')

