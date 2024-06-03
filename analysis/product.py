import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analysis import Analysis


class Product(Analysis):

    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.cross_tab_product_normalized = None
        self.alcohol_products = None

    def _analyze(self):
        product_department_tip = self.orders_joined[['product_id', 'department_id', 'tip']]
        product_mapping = self.products.set_index('product_id')['product_name']

        alcohol_department_id = self.departments[self.departments['department'] == 'alcohol']['department_id'].values[0]
        alcohol_product_ids = product_department_tip[product_department_tip['department_id'] == alcohol_department_id][
            'product_id'].unique()
        self.alcohol_products = product_mapping.loc[alcohol_product_ids].values

        self.cross_tab_product_normalized = (pd.crosstab(index=product_department_tip['product_id'],
                                                         columns=product_department_tip['tip'],
                                                         margins=True,
                                                         normalize='index')
                                             .rename(index=product_mapping)
                                             .sort_values(by=1, ascending=False))

    def _show_results(self):
        self._plot_cross_tab(self.cross_tab_product_normalized, 'Product', self.alcohol_products,
                             'Alcohol')

    def _plot_cross_tab(self, cross_tab_normalized, feature, highlighted_indices, highlighted_name):
        fig, ax2 = plt.subplots(1, 1, figsize=(22, 6))  # 1 row, 2 columns

        tip_probabilities = cross_tab_normalized[cross_tab_normalized.index != 'All'][1]
        mean_probability = cross_tab_normalized[1]['All']

        colors = ['C1' if idx in highlighted_indices else 'C0' for idx in tip_probabilities.index]

        # Subplot for Probability
        ax2.bar(tip_probabilities.index, tip_probabilities, color=colors, label='Tip Probability')
        ax2.axhline(y=mean_probability, linestyle='--', label='Mean Tip Probability', color='red')
        # ax2.set_xticks(tip_probabilities.index)
        # ax2.set_xticklabels(tip_probabilities.index, rotation=90, ha="right")
        ax2.set_xlabel(feature)
        ax2.set_ylabel("Tip Probability")
        ax2.set_title(f"Tip Probability by {feature}")

        legend_elements = [
            Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Mean Tip Probability'),
            Patch(facecolor='C1', label=f'{highlighted_name} Products'),
            Patch(facecolor='C0', label=f'Non-{highlighted_name} Products')]
        ax2.legend(handles=legend_elements)

        # Show the plot
        plt.tight_layout()
        plt.show()

        # if save_plots:
        #     self._save_plot(fig, 'aisle.png')
