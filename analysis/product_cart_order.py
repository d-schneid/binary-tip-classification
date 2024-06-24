import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
import math
from analysis import Analysis


class ProductCartOrder(Analysis):

    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.product_card_order_tip_rate = None
        self.product_card_normalized = None
        self.product_tip_standard_deviation = None

    def _analyze(self):
        # plot 1
        orders_joined_card_norm = self.orders_joined.copy()

        orders_joined_card_norm['normalized_add_to_cart_order'] = \
            (orders_joined_card_norm.groupby('order_id')['add_to_cart_order']
             .transform(lambda x: (x - x.min()) / (x.max() - x.min())))

        bins = np.arange(0, 1.1, 0.1)
        labels = range(1, 11)

        orders_joined_card_norm['binned_normalized_add_to_cart_order'] = (
            pd.cut(orders_joined_card_norm['normalized_add_to_cart_order'],
                   bins=bins, labels=labels, include_lowest=True))

        self.product_card_order_tip_rate = (
            orders_joined_card_norm.groupby(['product_id', 'binned_normalized_add_to_cart_order'], observed=True)['tip']
            .mean().reset_index())

        # plot 2
        #product_card_order_tip_rate_first_bin = orders_joined_card_norm[
        #    orders_joined_card_norm['binned_normalized_add_to_cart_order'] == 0.05]

        #self.product_card_normalized = (pd.crosstab(index=product_card_order_tip_rate_first_bin['product_id'],
        #                                            columns=product_card_order_tip_rate_first_bin['tip'],
        #                                            margins=True,
        #                                            normalize='index').sort_values(by=1, ascending=False))

        # plot 3 - standard derivation
        #product_tip_rate = orders_joined_card_norm.groupby('product_id')['tip'].mean().reset_index()
        #product_tip_rate.columns = ['product_id', 'product_tip_rate']

        #product_card_order_tip_rates = pd.merge(self.product_card_order_tip_rate, product_tip_rate, on='product_id',
        #                                        how='left')

        #self.product_tip_standard_deviation = product_card_order_tip_rates.groupby('product_id').apply(
        #    self._calculate_variance, include_groups=False).reset_index()

        #self.product_tip_standard_deviation.columns = ['product_id', 'product_tip_bin_variance']

    def _show_results(self, save_plots=False):
        self._plot_cross_tab(self.product_card_order_tip_rate, self.product_tip_standard_deviation, save_plots)
        # self._plot_first_bin_tip_rate(self.product_card_normalized)

    def _calculate_variance(self, product_bins):
        sum_of_distance_to_mean = 0

        for index, bin_value in product_bins.iterrows():
            sum_of_distance_to_mean += math.pow(bin_value['tip'] - bin_value['product_tip_rate'], 2)

        return math.sqrt(sum_of_distance_to_mean / len(product_bins))

    def _plot_cross_tab(self, product_card_order_tip_rate, product_tip_standard_deviation, save_plots=False):
        fig, ax = plt.subplots(1, 1, figsize=(20, 6))

        # Figure 1 Boxplot
        tip_probability_per_bin = product_card_order_tip_rate
        sns.boxplot(data=tip_probability_per_bin, x='binned_normalized_add_to_cart_order', y='tip', ax=ax)
        ax.set_xlabel('Normalized Add to Cart Order')
        ax.set_ylabel('Distribution of Tip Probabilities')
        ax.set_title('Tip Probability Distribution by Normalized Add to Cart Order')

        # Figure 2 Standard Deviation Histogram
        # ax2.hist(product_tip_standard_deviation['product_tip_bin_variance'], bins=20, color='C0', edgecolor='black', alpha=0.7)
        # ax2.set_xlabel('Normalized Add to Cart Order standard deviation')
        # ax2.set_ylabel('Amount of products')
        # ax2.set_title('Distribution of standard deviation')

        # Show the plot
        plt.tight_layout()
        plt.show()

        if save_plots:
            self._save_plot(fig, 'product_card_order.png')

    def _plot_first_bin_tip_rate(self, cross_tab_normalized):
        fig, ax = plt.subplots(1, 1, figsize=(20, 6))

        tip_probabilities = cross_tab_normalized[cross_tab_normalized.index != 'All'][1]
        mean_probability = cross_tab_normalized[1]['All']

        ax.hist(tip_probabilities, bins=20, color='C0', edgecolor='black', alpha=0.7)

        ax.axvline(x=mean_probability, linestyle='--', label='Mean Tip Probability', color='red')
        ax.set_xlabel('Tip Probability')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Tip Probabilities in first bin of normalized order')

        legend_elements = [
            Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Mean Tip Probability')]
        ax.legend(handles=legend_elements)
