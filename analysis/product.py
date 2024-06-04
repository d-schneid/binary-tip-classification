import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analysis import Analysis


class Product(Analysis):

    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.product_probability_frequency = None
        self.alcohol_products = None
        self.mean_tip_probability = None
        self.mean_order_frequency = None
        self.median_order_frequency = None

    def _analyze(self):
        product_department_tip = self.orders_joined[['product_id', 'department_id', 'tip', 'product_name']]
        self.alcohol_products = product_department_tip[product_department_tip['department_id'] == 5][
            'product_id'].unique()

        cross_tab_product = pd.crosstab(index=product_department_tip['product_id'],
                                        columns=product_department_tip['tip'],
                                        margins=True)

        self.mean_order_frequency = cross_tab_product['All'].mean()
        self.median_order_frequency = cross_tab_product['All'].median()

        cross_tab_product_normalized = pd.crosstab(index=product_department_tip['product_id'],
                                                   columns=product_department_tip['tip'],
                                                   margins=True,
                                                   normalize='index')

        self.mean_tip_probability = cross_tab_product_normalized[1]['All']

        cross_tab_product_normalized = cross_tab_product_normalized.drop('All', axis=0)
        cross_tab_product = cross_tab_product.drop('All', axis=0)

        self.product_probability_freq = pd.merge(cross_tab_product_normalized, cross_tab_product['All'],
                                                 left_index=True,
                                                 right_index=True)
        self.product_probability_freq = pd.merge(self.product_probability_freq,
                                                 self.products[['product_id', 'product_name', 'department_id']],
                                                 left_index=True,
                                                 right_on='product_id')

    def _show_results(self):
        self._plot_distribution(self.product_probability_freq, weighted=False)
        self._plot_distribution(self.product_probability_freq, weighted=True)
        self._print_general_facts()
        self._plot_partial_distribution(self.product_probability_freq, 0.8, 1.0, 20)
        self._print_upper_bound_statistics()
        self._plot_partial_distribution(self.product_probability_freq, 0.0, 0.2, 20)
        self._print_lower_bound_statistics()
        self._plot_no_tip_distribution(self.product_probability_freq)

    def _plot_distribution(self, product_probability_freq, weighted):
        fig, ax = plt.subplots(1, 1, figsize=(20, 6))

        alcohol_products = product_probability_freq[product_probability_freq['department_id'] == 5]
        non_alcohol_products = product_probability_freq[product_probability_freq['department_id'] != 5]

        if weighted:
            ax.hist([alcohol_products[1], non_alcohol_products[1]],
                    bins=100,
                    weights=[alcohol_products['All'], non_alcohol_products['All']],
                    stacked=True,
                    edgecolor='white',
                    color=['C1', 'C0'])
            ax.set_title('Distribution of Tip Rates by Product (Weighted by Order Frequency)')
            ax.set_ylabel('Product Order Frequency')
        else:
            ax.hist([alcohol_products[1], non_alcohol_products[1]],
                    bins=100,
                    edgecolor='white',
                    stacked=True,
                    color=['C1', 'C0'])
            ax.set_title('Distribution of Tip Rates by Product')
            ax.set_ylabel('Product Frequency')

        ax.set_xlabel('Tip Rate')
        ax.axvline(self.mean_tip_probability,
                   color='red',
                   linestyle='dotted')

        legend_elements = [
            Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Mean Tip Rate'),
            Patch(facecolor='C1', label=f'Alcohol Products'),
            Patch(facecolor='C0', label=f'Non-Alcohol Products')]

        ax.legend(handles=legend_elements)
        plt.tight_layout()
        plt.show()

    def _plot_partial_distribution(self, product_probability_freq, lower_bound, upper_bound, bins):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

        range_mask = (product_probability_freq[1] >= lower_bound) & (product_probability_freq[1] <= upper_bound)
        alcohol_mask = product_probability_freq['department_id'] == 5

        alcohol_products = product_probability_freq[alcohol_mask & range_mask]
        non_alcohol_products = product_probability_freq[~alcohol_mask & range_mask]

        ax1.hist([alcohol_products[1], non_alcohol_products[1]],
                 bins=bins,
                 stacked=True,
                 edgecolor='white',
                 color=['C1', 'C0'])

        ax1.set_title(f'Products with Tip Rate between {lower_bound * 100:.0f}% and {upper_bound * 100:.0f}%')
        ax1.set_xlabel('Tip Rate')
        ax1.set_ylabel('Product Frequency')

        ax2.hist([alcohol_products[1], non_alcohol_products[1]],
                 bins=bins,
                 stacked=True,
                 weights=[alcohol_products['All'], non_alcohol_products['All']],
                 edgecolor='white',
                 color=['C1', 'C0'])

        ax2.set_title(
            f'Products with Tip Rate between {lower_bound * 100:.0f}% and {upper_bound * 100:.0f}% (Weighted by Order Frequency)')
        ax2.set_xlabel('Tip Rate')
        ax2.set_ylabel('Product Order Frequency')

        legend_elements = [
            Patch(facecolor='C1', label=f'Alcohol Products'),
            Patch(facecolor='C0', label=f'Non-Alcohol Products')]
        ax1.legend(handles=legend_elements)
        ax2.legend(handles=legend_elements)

        plt.tight_layout()
        plt.show()

    def _plot_no_tip_distribution(self, product_probability_freq):
        fig, ax = plt.subplots(1, 1, figsize=(20, 6))

        no_tip_products = product_probability_freq[product_probability_freq[1] == 0.0][['All', 'product_id']].groupby(
            'All').count()

        ax.bar(no_tip_products.index, no_tip_products['product_id'], edgecolor='white', label='No Tip Products')
        ax.set_xticks(no_tip_products.index)

        ax.set_xlabel('Order Frequency')
        ax.set_ylabel('Number of Products')
        ax.set_title('Number of Products with 0% Tip Rate by Order Frequency')

        ax.legend()
        plt.tight_layout()
        plt.show()

    def _print_general_facts(self):
        print(f"Mean Order Frequency: {self.mean_order_frequency:.2f}")
        print(f"Median Order Frequency: {self.median_order_frequency:.2f}")

    def _print_upper_bound_statistics(self):
        print("Top 5 Non-Alcohol Products with 100% Tip Rate sorted by Order Frequency")
        print(self.product_probability_freq[
                  (self.product_probability_freq[1] == 1.0) & (self.product_probability_freq['department_id'] != 5)]
              .sort_values(by='All', ascending=False)
              .head(5)[['product_name', 'All']]
              .rename(columns={'All': 'Order Frequency',
                               'product_name': 'Product Name'}))

    def _print_lower_bound_statistics(self):
        print("Top 5 Products with 0% Tip Rate sorted by Order Frequency")
        print(self.product_probability_freq[self.product_probability_freq[1] == 0.0]
              .sort_values(by='All', ascending=False)
              .head(5)[['product_name', 'All']]
              .rename(columns={'All': 'Order Frequency',
                               'product_name': 'Product Name'}))
