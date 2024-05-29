import pandas as pd
from matplotlib import pyplot as plt

from analysis import Analysis


class OrderContents(Analysis):

    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.cross_tab_department = None
        self.cross_tab_department_normalized = None

    def _analyze(self):
        prod_aisle_dep_tip = self.orders_joined[['product_id', 'aisle_id', 'department_id', 'tip']]
        department_mapping = self.departments.set_index('department_id')['department']

        self.cross_tab_department = pd.crosstab(index=prod_aisle_dep_tip['department_id'],
                                                columns=prod_aisle_dep_tip['tip'],
                                                margins=True).rename(index=department_mapping)

        self.cross_tab_department_normalized = pd.crosstab(index=prod_aisle_dep_tip['department_id'],
                                                           columns=prod_aisle_dep_tip['tip'],
                                                           margins=True,
                                                           normalize='index').rename(index=department_mapping)

    def _show_results(self):
        self._plot_cross_tab(self.cross_tab_department, self.cross_tab_department_normalized, 'Department')

    def _plot_cross_tab(self, cross_tab, cross_tab_normalized, feature):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))  # 1 row, 2 columns

        # Subplot for Frequency
        no_tip_data = cross_tab[0][:-1]
        tip_data = cross_tab[1][:-1]
        ax1.bar(cross_tab.index[:-1], tip_data, color='green', label='Tip', alpha=0.5)
        ax1.bar(cross_tab.index[:-1], no_tip_data, color='red', label='No Tip', alpha=0.5)
        ax1.set_xlabel(feature)
        ax1.set_ylabel("Order Frequency")
        ax1.set_title(f"Frequency of Orders by {feature}")
        ax1.legend()

        # Subplot for Probability
        mean_probability = cross_tab_normalized[1]['All']
        # midpoint = (cross_tab_normalized.index[:-1].astype(int).min() + cross_tab_normalized.index[:-1].astype(
        #     int).max()) / 2

        ax2.bar(cross_tab_normalized.index[:-1], cross_tab_normalized[1][:-1],
                label='Tip Probability')
        ax2.axhline(y=mean_probability, linestyle='--', label='Mean Tip Probability', color='red')
        # ax2.text(x=midpoint, y=mean_probability, s=f'Mean: {mean_probability:.2f}', color='red',
        #          va='bottom', ha='right')
        ax2.set_xlabel(feature)
        ax2.set_ylabel("Tip Probability")
        ax2.set_title(f"Tip Probability by {feature}")
        ax2.legend()

        # Show the plot
        plt.tight_layout()
        plt.show()
