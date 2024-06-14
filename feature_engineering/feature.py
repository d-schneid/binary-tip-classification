from abc import ABC, abstractmethod
import seaborn as sns  #
import pandas as pd
from matplotlib import pyplot as plt


class Feature(ABC):

    def __init__(self, name):
        self.orders_tip = None
        self.orders_joined = None
        self.feature = name

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
        tip = orders_tip_features['tip']
        feature_data = orders_tip_features[self.feature]
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Print Correlation
        feature_tip_correlation = feature_data.corr(tip)
        print(f'Correlation between {self.feature} and tip: {feature_tip_correlation}')

        # Violinplot
        sns.violinplot(data=orders_tip_features, x='tip', y=self.feature, ax=ax1)
        ax1.set_title(f'{self.feature} vs tip')
        ax1.set_xlabel('tip')
        ax1.set_ylabel(self.feature)

        unique_values = orders_tip_features[self.feature].unique()
        tip_data = orders_tip_features[orders_tip_features['tip'] == 1]
        no_tip_data = orders_tip_features[orders_tip_features['tip'] == 0]
        if len(unique_values) > 2:
            # linegraph
            sns.kdeplot(tip_data[self.feature],
                        linewidth=3,
                        label='survived',
                        color='green',
                        ax=ax2
                        )
            sns.kdeplot(no_tip_data[self.feature],
                        linewidth=3,
                        label='survived',
                        color='red',
                        ax=ax2)
            ax2.set_title(f'{self.feature} vs density of tip & no tip')
        else:
            cross_tab = pd.crosstab(index=orders_tip_features[self.feature],
                                    columns=orders_tip_features['tip'],
                                    margins=True)
            no_tip_data = cross_tab[0][:-1]
            tip_data = cross_tab[1][:-1]
            ax2.bar(cross_tab.index[:-1].astype(int), tip_data, color='green', label='Tip', alpha=0.5)
            ax2.bar(cross_tab.index[:-1].astype(int), no_tip_data, color='red', label='No Tip', alpha=0.5)
            ax2.set_xlabel(self.feature)
            ax2.set_ylabel("Order Frequency")
            ax2.set_title(f"Frequency of Orders by {self.feature}")
            ax2.legend()

        sns.histplot(data=orders_tip_features, x=self.feature, hue='tip', bins=10, element='step', stat='density', ax=ax3,
                     common_norm=False)
        ax3.set_title(f'{self.feature} vs density of tip & no tip')

        plt.tight_layout()
        plt.show()

    @abstractmethod
    def _analyze_feature(self):
        pass


class StaticFeature(Feature):

    def __init__(self, name):
        super().__init__(name)

    @abstractmethod
    def _compute_feature(self):
        pass

    @abstractmethod
    def _analyze_feature(self):
        pass


class DynamicFeature(Feature):

    def __init__(self, name):
        super().__init__(name)

    @abstractmethod
    def _compute_feature(self):
        pass

    @abstractmethod
    def _analyze_feature(self):
        pass
