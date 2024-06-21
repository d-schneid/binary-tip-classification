from abc import ABC, abstractmethod
import seaborn as sns  #
import pandas as pd
from matplotlib import pyplot as plt


class Feature(ABC):

    def __init__(self, name):
        self.orders_tip = None
        self.orders_joined = None
        self.feature = name
        self.feature_type = None
        self.BINARY_FEATURE = 'binary'
        self.DISCRETE_FEATURE = 'discrete'
        self.STEADY_FEATURE = 'steady'

    def compute_feature(self):
        self._handle_missing_values()
        self._compute_feature()

    def _handle_missing_values(self):
        # self.orders_tip['days_since_prior_order'] = self.orders_tip['days_since_prior_order'].fillna(-1).astype(int)
        pass

    def get_feature_name(self):
        return self.feature

    def set_orders_tip(self, orders_tip):
        self.orders_tip = orders_tip

    def set_orders_joined(self, orders_joined):
        self.orders_joined = orders_joined

    def get_orders_tip(self):
        return self.orders_tip

    def get_orders_joined(self):
        return self.orders_joined

    @abstractmethod
    def _compute_feature(self):
        pass

    def analyze_feature(self, orders_tip_features):
        self._analyze_feature(orders_tip_features)

    def _analyze_feature(self, orders_tip_features):
        tip = orders_tip_features['tip']
        feature_data = orders_tip_features[self.feature]

        feature_tip_correlation = feature_data.corr(tip)
        print(f'Correlation between {self.feature} and tip: {feature_tip_correlation}')

        if self.feature_type == self.BINARY_FEATURE:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            self._create_violin_plot(orders_tip_features, ax1)
            self._create_box_plot(orders_tip_features, ax2)
        elif self.feature_type == self.STEADY_FEATURE:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            self._create_violin_plot(orders_tip_features, ax1)
            self._create_density_plot(orders_tip_features, ax2)
            self._create_tip_rate_plot_steady(orders_tip_features, ax3)
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            self._create_violin_plot(orders_tip_features, ax1)
            self._create_density_plot(orders_tip_features, ax2)
            self._create_tip_rate_plot_discrete(orders_tip_features, ax3)

        plt.tight_layout()
        plt.show()

    def _create_violin_plot(self, orders_tip_features, ax):
        sns.violinplot(data=orders_tip_features, x='tip', y=self.feature, ax=ax)
        ax.set_title(f'{self.feature} vs tip')
        ax.set_xlabel('tip')
        ax.set_ylabel(self.feature)

    def _create_box_plot(self, orders_tip_features, ax):
        feature_tip_rate = orders_tip_features.groupby(self.feature)['tip'].mean().reset_index()
        feature_tip_rate.columns = [self.feature, 'tip_rate']

        sns.histplot(data=orders_tip_features, x=self.feature, hue='tip', bins=10, element='step', stat='density',
                     ax=ax,
                     common_norm=False)
        ax.set_xlabel(self.feature)
        ax.set_ylabel('tip rate')
        ax.set_title(f'{self.feature} vs tip rate')

    def _create_density_plot(self, orders_tip_features, ax):
        tip_data = orders_tip_features[orders_tip_features['tip'] == 1]
        no_tip_data = orders_tip_features[orders_tip_features['tip'] == 0]
        sns.kdeplot(tip_data[self.feature],
                    linewidth=3,
                    label='survived',
                    color='green',
                    ax=ax
                    )
        sns.kdeplot(no_tip_data[self.feature],
                    linewidth=3,
                    label='survived',
                    color='red',
                    ax=ax)
        ax.set_title(f'{self.feature} vs density of tip & no tip')

    def _create_tip_rate_plot_steady(self, orders_tip_features, ax):
        orders_tip_features = orders_tip_features.copy()
        num_bins = 10
        orders_tip_features['binned_feature'] = pd.cut(orders_tip_features[self.feature], bins=num_bins)
        feature_tip_rate = orders_tip_features.groupby('binned_feature', observed=True)['tip'].mean().reset_index()
        feature_tip_rate.columns = ['binned_feature', 'tip_rate']
        feature_tip_rate['feature_bin_mid'] = feature_tip_rate['binned_feature'].apply(lambda x: x.mid)

        sns.lineplot(data=feature_tip_rate,
                     x='feature_bin_mid',
                     y='tip_rate',
                     marker='o',
                     ax=ax)

        ax.set_xlabel(self.feature)
        ax.set_ylabel('tip rate')
        ax.set_title(f'{self.feature} vs tip rate')

    def _create_tip_rate_plot_discrete(self, orders_tip_features, ax):
        feature_tip_rate = orders_tip_features.groupby(self.feature)['tip'].mean().reset_index()
        feature_tip_rate.columns = [self.feature, 'tip_rate']

        sns.lineplot(data=feature_tip_rate,
                     x=self.feature,
                     y='tip_rate',
                     ax=ax)
        ax.set_xlabel(self.feature)
        ax.set_ylabel('tip rate')
        ax.set_title(f'{self.feature} vs tip rate')


class StaticFeature(Feature):

    def __init__(self, name):
        super().__init__(name)

    @abstractmethod
    def _compute_feature(self):
        pass


class DynamicFeature(Feature):

    def __init__(self, name):
        super().__init__(name)

    @abstractmethod
    def _compute_feature(self):
        pass
