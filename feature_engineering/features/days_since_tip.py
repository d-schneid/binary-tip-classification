import numpy as np
import pandas as pd

from feature_engineering import StaticFeature


class DaysSinceTip(StaticFeature):

	def __init__(self, data_store):
		super().__init__(data_store, 'days_since_tip')

	def _compute_feature(self):
		orders_tip_copy = self.orders_tip[['order_id', 'user_id', 'order_number', 'days_since_prior_order', 'tip']].copy()
		# set to NaN for accurate computation, since it is set to -1 by superclass beforehand
		orders_tip_copy.loc[self.orders_tip['order_number'] == 1, 'days_since_prior_order'] = np.nan
		orders_tip_copy['cum_days'] = orders_tip_copy.groupby('user_id')['days_since_prior_order'].cumsum()

		# cumulative days of orders with tip as reference points
		orders_tip_copy['cum_days_last_tip'] = np.where(orders_tip_copy['tip'] == 1, orders_tip_copy['cum_days'], np.nan)
		orders_tip_copy.loc[orders_tip_copy['order_number'] == 1, 'cum_days_last_tip'] = -1
		# handle edge case, otherwise -1 of first order would propagate to following orders
		orders_tip_copy.loc[(orders_tip_copy['order_number'] == 1) & (orders_tip_copy['tip'] == 1), 'cum_days_last_tip'] = 0

		# assign all orders their respective reference point (cumulative days of last order with tip)
		orders_tip_copy.loc[:, 'cum_days_last_tip'] = orders_tip_copy.loc[:, 'cum_days_last_tip'].ffill().shift(1)
		# no previous order with tip for first order
		orders_tip_copy.loc[orders_tip_copy['order_number'] == 1, 'cum_days_last_tip'] = -1
		# difference between cumulative days of order and its reference point (cumulative days of last order with tip)
		# is the number of days since the last order with a tip occurred
		orders_tip_copy[self.feature] = np.where(orders_tip_copy['cum_days_last_tip'] != -1,
												 orders_tip_copy['cum_days'] - orders_tip_copy['cum_days_last_tip'], -1)

		self.orders_tip = pd.merge(self.orders_tip, orders_tip_copy[['order_id', self.feature]], on='order_id', how='left')

	def _analyze_feature(self):
		pass
