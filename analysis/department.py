import pandas as pd
from matplotlib import pyplot as plt

from analysis import Analysis


class Department(Analysis):

    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.cross_tab_department = None
        self.cross_tab_department_normalized = None

    def _analyze(self):
        department_tip = self.orders_joined[['department_id', 'tip']]
        department_mapping = self.departments.set_index('department_id')['department']

        self.cross_tab_department = pd.crosstab(index=department_tip['department_id'],
                                                columns=department_tip['tip'],
                                                margins=True).rename(index=department_mapping)

        self.cross_tab_department_normalized = (pd.crosstab(index=department_tip['department_id'],
                                                            columns=department_tip['tip'],
                                                            margins=True,
                                                            normalize='index')
                                                .rename(index=department_mapping)
                                                .sort_values(by=1, ascending=False))

    def _show_results(self, save_plots=False):
        self._plot_cross_tab(self.cross_tab_department, self.cross_tab_department_normalized, 'Department',
                             save_plots=save_plots)

    def _plot_cross_tab(self, cross_tab, cross_tab_normalized, feature, save_plots=False):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))  # 1 row, 2 columns

        tip_probabilities = cross_tab_normalized[cross_tab_normalized.index != 'All'][1]
        mean_probability = cross_tab_normalized[1]['All']

        cross_tab = cross_tab.drop('All', axis=0).reindex(tip_probabilities.index)

        # Subplot for Frequency
        no_tip_data = cross_tab[0]
        tip_data = cross_tab[1]
        ax1.bar(cross_tab.index, tip_data, color='green', label='Tip', alpha=0.5)
        ax1.bar(cross_tab.index, no_tip_data, color='red', label='No Tip', alpha=0.5)
        ax1.set_xticks(cross_tab.index)
        ax1.set_xticklabels(cross_tab.index, rotation=45, ha="right")
        ax1.set_xlabel(feature)
        ax1.set_ylabel("Order Frequency")
        ax1.set_title(f"Frequency of Orders by {feature}")
        ax1.legend()

        # Subplot for Probability
        ax2.bar(tip_probabilities.index, tip_probabilities,
                label='Tip Probability')
        ax2.axhline(y=mean_probability, linestyle='--', label='Mean Tip Probability', color='red')
        ax2.set_xticks(tip_probabilities.index)
        ax2.set_xticklabels(tip_probabilities.index, rotation=45, ha="right")
        ax2.set_xlabel(feature)
        ax2.set_ylabel("Tip Probability")
        ax2.set_title(f"Tip Probability by {feature}")
        ax2.legend()

        # Show the plot
        plt.tight_layout()
        plt.show()

        if save_plots:
            self._save_plot(fig, 'department.png')
