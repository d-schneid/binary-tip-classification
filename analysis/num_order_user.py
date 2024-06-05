import pandas as pd
from matplotlib import pyplot as plt

from analysis import Analysis


class NumberOrderUser(Analysis):

    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.df_rate = None
        self.df_freq = None

    def _analyze(self):
        ot = self.orders_tip
        order_volume = ot.groupby('user_id').count()['order_id'].rename('order_volume').reset_index()
        ot = pd.merge(ot, order_volume, on='user_id', how='left')

        tip_rate = (ot.groupby('user_id').sum()['tip'] / ot.groupby('user_id').count()['tip']).rename(
            'tip_rate').reset_index()
        ot = pd.merge(ot, tip_rate, on='user_id', how='left')

        ot = ot.drop_duplicates(subset='user_id')[['user_id', 'order_volume', 'tip_rate']]

        self.df_rate = ot.groupby('order_volume').mean().reset_index()[['order_volume', 'tip_rate']]
        self.df_freq = (ot.groupby('order_volume').count().reset_index()[['order_volume', 'tip_rate']]
                        .rename(columns={'tip_rate': 'frequency'}))

    def _show_results(self, save_plots=False):
        self._plot_fig(self.df_rate, self.df_freq, save_plots=save_plots)

    def _plot_fig(self, df_rate, df_freq, save_plots=False):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))

        ax1.plot(df_rate['order_volume'], df_rate['tip_rate'], marker='o', linestyle='-', color='C0')
        ax1.set_xlabel('Order Volume')
        ax1.set_ylabel('Mean Tip Rate')
        ax1.set_title('Mean Tip Rate by Order Volume')

        ax2.plot(df_freq['order_volume'], df_freq['frequency'], marker='o', linestyle='-', color='C1')
        ax2.set_xlabel('Order Volume')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Frequency of Order Volume')

        plt.tight_layout()
        plt.show()

        if save_plots:
            self._save_plot(fig, 'NumOrderUser.png')
