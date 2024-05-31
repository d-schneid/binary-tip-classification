import pandas as pd
from matplotlib import pyplot as plt

from analysis import Analysis


class Temporary(Analysis):

    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.cross_tab_dspo = None
        self.cross_tab_dspo_normalized = None
        self.cross_tab_on = None
        self.cross_tab_on_normalized = None

    def _analyze(self):
        dspo_on_tip = self.orders_tip[['order_number', 'days_since_prior_order', 'tip']]

        self.cross_tab_dspo = pd.crosstab(index=dspo_on_tip['days_since_prior_order'], columns=dspo_on_tip['tip'],
                                          margins=True)
        self.cross_tab_dspo_normalized = pd.crosstab(index=dspo_on_tip['days_since_prior_order'],
                                                     columns=dspo_on_tip['tip'],
                                                     margins=True,
                                                     normalize='index')

        self.cross_tab_on = pd.crosstab(index=dspo_on_tip['order_number'], columns=dspo_on_tip['tip'], margins=True)
        self.cross_tab_on_normalized = pd.crosstab(index=dspo_on_tip['order_number'], columns=dspo_on_tip['tip'],
                                                   margins=True,
                                                   normalize='index')

    def _show_results(self):
        self._plot_cross_tab(self.cross_tab_dspo, self.cross_tab_dspo_normalized, 'Days Since Prior Order')
        self._plot_cross_tab(self.cross_tab_on, self.cross_tab_on_normalized, 'Order Number')

    def _plot_cross_tab(self, cross_tab, cross_tab_normalized, feature):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))  # 1 row, 2 columns

        # Subplot for Frequency
        no_tip_data = cross_tab[0][:-1]
        tip_data = cross_tab[1][:-1]
        ax1.bar(cross_tab.index[:-1].astype(int), tip_data, color='green', label='Tip', alpha=0.5)
        ax1.bar(cross_tab.index[:-1].astype(int), no_tip_data, color='red', label='No Tip', alpha=0.5)
        ax1.set_xlabel(feature)
        ax1.set_ylabel("Order Frequency")
        ax1.set_title(f"Frequency of Orders by {feature}")
        ax1.legend()

        # Subplot for Probability
        mean_probability = cross_tab_normalized[1]['All']
        ax2.plot(cross_tab_normalized.index[:-1].astype(int), cross_tab_normalized[1][:-1], marker='o', linestyle='-',
                 label='Tip Probability')
        ax2.axhline(y=mean_probability, linestyle='--', color='red',
                    label=f'Mean Tip Probability')
        ax2.set_xlabel(feature)
        ax2.set_ylabel("Tip Probability")
        ax2.set_title(f"Tip Probability by {feature}")
        ax2.legend()

        mean_probability = cross_tab_normalized[1]['All']
        # midpoint = (cross_tab_normalized.index[:-1].astype(int).min() + cross_tab_normalized.index[:-1].astype(
        #     int).max()) / 2

        ax3.bar(cross_tab_normalized.index[:-1].astype(int), cross_tab_normalized[1][:-1],
                label='Tip Probability')
        ax3.axhline(y=mean_probability, linestyle='--', label='Mean Tip Probability', color='red')
        # ax2.text(x=midpoint, y=mean_probability, s=f'Mean: {mean_probability:.2f}', color='red',
        #          va='bottom', ha='right')
        ax3.set_xlabel(feature)
        ax3.set_ylabel("Tip Probability")
        ax3.set_title(f"Tip Probability by {feature}")
        ax3.legend()

        # Show the plot
        plt.tight_layout()
        plt.show()
