import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analysis import Analysis


class Aisle(Analysis):

    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.cross_tab_aisle_normalized = None
        self.alcohol_aisles = None

    def _analyze(self):
        prod_aisle_dep_tip = self.orders_joined[['product_id', 'aisle_id', 'department_id', 'tip']]
        aisle_mapping = self.aisles.set_index('aisle_id')['aisle']

        alcohol_department_id = self.departments[self.departments['department'] == 'alcohol']['department_id'].values[0]
        alcohol_aisle_ids = prod_aisle_dep_tip[prod_aisle_dep_tip['department_id'] == alcohol_department_id][
            'aisle_id'].unique()
        self.alcohol_aisles = aisle_mapping.loc[alcohol_aisle_ids].values

        self.cross_tab_aisle_normalized = (pd.crosstab(index=prod_aisle_dep_tip['aisle_id'],
                                                       columns=prod_aisle_dep_tip['tip'],
                                                       margins=True,
                                                       normalize='index')
                                           .rename(index=aisle_mapping)
                                           .sort_values(by=1, ascending=False))

    def _show_results(self):
        self._plot_cross_tab(self.cross_tab_aisle_normalized, 'Aisle', self.alcohol_aisles,
                             'Alcohol')

    def _plot_cross_tab(self, cross_tab_normalized, feature, highlighted_indices, highlighted_name):
        fig, ax2 = plt.subplots(1, 1, figsize=(20, 6))  # 1 row, 2 columns

        tip_probabilities = cross_tab_normalized[cross_tab_normalized.index != 'All'][1]
        mean_probability = cross_tab_normalized[1]['All']

        colors = ['C1' if idx in highlighted_indices else 'C0' for idx in tip_probabilities.index]

        # Subplot for Probability
        ax2.bar(tip_probabilities.index, tip_probabilities, color=colors, label='Tip Probability')
        ax2.axhline(y=mean_probability, linestyle='--', label='Mean Tip Probability', color='red')
        ax2.set_xticks(tip_probabilities.index)
        ax2.set_xticklabels(tip_probabilities.index, rotation=90, ha="right")
        ax2.set_xlabel(feature)
        ax2.set_ylabel("Tip Probability")
        ax2.set_title(f"Tip Probability by {feature}")

        legend_elements = [
            Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Mean Tip Probability'),
            Patch(facecolor='C1', label=f'{highlighted_name} Aisles'),
            Patch(facecolor='C0', label=f'Non-{highlighted_name} Aisles')]
        ax2.legend(handles=legend_elements)

        # Show the plot
        plt.tight_layout()
        plt.show()
