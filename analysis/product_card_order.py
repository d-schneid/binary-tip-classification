import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from analysis import Analysis


class ProductCardOrder(Analysis):

    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.product_card_order_tip_rate = None
        self.product_card_normalized = None

    def _analyze(self):
        # plot 1
        orders_joined_card_norm = self.orders_joined.copy()

        orders_joined_card_norm['normalized_add_to_cart_order'] = \
            (orders_joined_card_norm.groupby('order_id')['add_to_cart_order']
             .transform(lambda x: (x - x.min()) / (x.max() - x.min())))

        bins = np.arange(0, 1.1, 0.1)
        labels = np.arange(0.05, 1.05, 0.1)

        orders_joined_card_norm['binned_normalized_add_to_cart_order'] = (
            pd.cut(orders_joined_card_norm['normalized_add_to_cart_order'],
                   bins=bins, labels=labels, include_lowest=True))

        self.product_card_order_tip_rate = (
            orders_joined_card_norm.groupby(['product_id', 'binned_normalized_add_to_cart_order'])['tip'].mean()
            .reset_index())

        # plot 2
        product_card_order_tip_rate_first_bin = orders_joined_card_norm[
            orders_joined_card_norm['binned_normalized_add_to_cart_order'] == 0.05]

        self.product_card_normalized = (pd.crosstab(index=product_card_order_tip_rate_first_bin['product_id'],
                                                    columns=product_card_order_tip_rate_first_bin['tip'],
                                                    margins=True,
                                                    normalize='index').sort_values(by=1, ascending=False))

    def _show_results(self):
        self._plot_cross_tab(self.product_card_normalized)
        self._plot_cross_tab(self.product_card_order_tip_rate)

    def _plot_boxplot(self, product_card_order_tip_rate):
        tip_probability_per_bin = product_card_order_tip_rate(columns='binned_normalized_add_to_cart_order',
                                                              values='tip')

        plt.figure(figsize=(14, 7))
        sns.boxplot(data=tip_probability_per_bin)
        plt.xlabel('Binned Normalized Add to Cart Order')
        plt.ylabel('Tip Probability')
        plt.title('Tip Probability Distribution by Binned Normalized Add to Cart Order')
        plt.show()

    def _plot_cross_tab(self, cross_tab_normalized):
        fig, ax2 = plt.subplots(1, 1, figsize=(22, 6))

        tip_probabilities = cross_tab_normalized[cross_tab_normalized.index != 'All'][1]
        mean_probability = cross_tab_normalized[1]['All']

        colors = ['C0' for idx in tip_probabilities.index]

        # Subplot for Probability
        ax2.bar(tip_probabilities.index, tip_probabilities, color=colors, label='Tip Probability')
        ax2.axhline(y=mean_probability, linestyle='--', label='Mean Tip Probability', color='red')
        # ax2.set_xticks(tip_probabilities.index)
        # ax2.set_xticklabels(tip_probabilities.index, rotation=90, ha="right")
        ax2.set_xlabel('Product')
        ax2.set_ylabel("Tip Probability")
        ax2.set_title(f"Tip Probability by Product with normalized card order")

        legend_elements = [
            Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Mean Tip Probability')]
        ax2.legend(handles=legend_elements)

        # Show the plot
        plt.tight_layout()
        plt.show()
