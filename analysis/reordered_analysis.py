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

    def _show_results(self):
        self._plot_scatter(self.tip_probabilities, self.reorder_probabilities)
        self._plot_scatter_orders(self.orders_joined)

    def _plot_scatter(self, tip_probabilities, reorder_probabilities, isOrderCountsConsidered=False):
        fig, ax = plt.subplots(figsize=(12, 8))

        if isOrderCountsConsidered:
            order_counts_per_product = self.orders_joined.groupby('product_id').size()

            # normalize order_counts for color allocation
            normalized_counts = ((order_counts_per_product - order_counts_per_product.min())
                                 / (order_counts_per_product.max() - order_counts_per_product.min()))

            # create scatter plot with color scale
            scatter = ax.scatter(reorder_probabilities, tip_probabilities, s=order_counts_per_product,
                                 c=normalized_counts, alpha=0.6, edgecolors='w', cmap='viridis')

            # Create color scale
            cbar = plt.colorbar(scatter)
            cbar.set_label('Normalized Order Count')

        else:
            scatter = ax.scatter(reorder_probabilities, tip_probabilities, alpha=0.6, edgecolors='w')

        ax.set_xlabel('Reorder Probability')
        ax.set_ylabel('Tip Probability')
        ax.set_title('Tip Probability vs Reorder Probability by Product')

        plt.show()

    def _plot_scatter_orders(self, df):
        tip_probabilities = df.groupby('product_id')['tip'].mean()
        reorder_probabilities = df.groupby('product_id')['reordered'].mean()
        order_counts = df.groupby('product_id').size()

        normalized_counts = (order_counts - order_counts.min()) / (order_counts.max() - order_counts.min())

        fig, ax = plt.subplots(figsize=(12, 8))

        scatter = ax.scatter(reorder_probabilities, tip_probabilities, s=order_counts, c=normalized_counts, alpha=0.6,
                             edgecolors='w', cmap='viridis')

        ax.set_xlabel('Reorder Probability')
        ax.set_ylabel('Tip Probability')
        ax.set_title('Tip Probability vs Reorder Probability by Product')

        cbar = plt.colorbar(scatter)
        cbar.set_label('Normalized Order Count')

        plt.show()
