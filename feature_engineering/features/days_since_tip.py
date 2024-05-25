import numpy as np
import pandas as pd

from feature_engineering import StaticFeature


class DaysSinceTip(StaticFeature):

	def __init__(self, data_store, name='days_since_tip'):
		super().__init__(data_store, name)
		self._feature = 'days_since_tip'

	def _compute_feature(self):
		orders_tip_copy = self._compute_days_since_tip()
		self.orders_tip = pd.merge(self.orders_tip, orders_tip_copy[['order_id', self._feature]], on='order_id', how='left')

	def _compute_days_since_tip(self):
		orders_tip_copy = self.orders_tip[
			['order_id', 'user_id', 'order_number', 'days_since_prior_order', 'tip']].copy()
		# set to NaN for accurate computation, since it is set to -1 by superclass beforehand
		orders_tip_copy.loc[self.orders_tip['order_number'] == 1, 'days_since_prior_order'] = np.nan
		orders_tip_copy['cum_days'] = orders_tip_copy.groupby('user_id')['days_since_prior_order'].cumsum()

		# cumulative days of orders with tip as reference points
		orders_tip_copy['cum_days_last_tip'] = np.where(orders_tip_copy['tip'] == 1, orders_tip_copy['cum_days'],
														np.nan)
		orders_tip_copy.loc[orders_tip_copy['order_number'] == 1, 'cum_days_last_tip'] = -1
		# handle edge case, otherwise -1 of first order would propagate to following orders
		orders_tip_copy.loc[
			(orders_tip_copy['order_number'] == 1) & (orders_tip_copy['tip'] == 1), 'cum_days_last_tip'] = 0

		# assign all orders their respective reference point (cumulative days of last order with tip)
		orders_tip_copy.loc[:, 'cum_days_last_tip'] = orders_tip_copy.loc[:, 'cum_days_last_tip'].ffill().shift(1)
		# no previous order with tip for first order
		orders_tip_copy.loc[orders_tip_copy['order_number'] == 1, 'cum_days_last_tip'] = -1
		# difference between cumulative days of order and its reference point (cumulative days of last order with tip)
		# is the number of days since the last order with a tip occurred
		orders_tip_copy[self._feature] = np.where(orders_tip_copy['cum_days_last_tip'] != -1,
												 orders_tip_copy['cum_days'] - orders_tip_copy['cum_days_last_tip'], -1).astype(int)
		return orders_tip_copy

	def _analyze_feature(self):
		pass


class RelDaysSinceTip(DaysSinceTip):

	def __init__(self, data_store):
		super().__init__(data_store, 'rel_days_since_tip')

	def _compute_feature(self):
		orders_tip_copy = self._compute_days_since_tip()
		orders_tip_copy['tip_temp'] = orders_tip_copy['tip'].fillna(0).astype(int)
		orders_tip_copy['num_tips'] = orders_tip_copy.groupby('user_id')['tip_temp'].cumsum()

		# prepare data for correct computation
		orders_tip_copy['days_since_tip_temp'] = orders_tip_copy['days_since_tip'].replace(-1, 0)
		orders_tip_copy.loc[orders_tip_copy['tip_temp'] == 0, 'days_since_tip_temp'] = 0

		# cumulative days only for orders where a tip was given
		orders_tip_copy['cum_days_tipped_orders'] = orders_tip_copy.groupby('user_id')['days_since_tip_temp'].cumsum()
		# use only days between two orders where for both a tip was given
		orders_tip_copy['mean_days_tipped_orders'] = np.where(orders_tip_copy['num_tips'] > 1,
															  orders_tip_copy['cum_days_tipped_orders'] / (orders_tip_copy['num_tips'] - 1),
															  -1)

		# shift to account for fact that for current order tip behavior is not known, since it shall be predicted
		# only previous orders along with their corresponding tipping behavior are known
		orders_tip_copy['mean_days_tipped_orders'] = orders_tip_copy['mean_days_tipped_orders'].shift(1)
		# for the first order there is no previous order (where a tip was given)
		orders_tip_copy.loc[orders_tip_copy['order_number'] == 1, 'mean_days_tipped_orders'] = -1

		# compute relative feature
		orders_tip_copy[self.feature] = np.where(orders_tip_copy['mean_days_tipped_orders'] != -1,
												 orders_tip_copy['days_since_tip'] - orders_tip_copy['mean_days_tipped_orders'],
												 np.nan)
		# scale feature into non-negative real numbers space to avoid conflicts with not useful feature values (-1)
		# Caution: This would convert feature to a dynamic feature!!
		# orders_tip_copy[self.feature] = orders_tip_copy[self.feature] - orders_tip_copy[self.feature].min(skipna=True)
		# set not useful feature values back to -1
		# orders_tip_copy.loc[pd.isna(orders_tip_copy[self.feature]), self.feature] = -1
		self.orders_tip = pd.merge(self.orders_tip, orders_tip_copy[['order_id', self.feature]], on='order_id', how='left')
