import numpy as np

from feature_engineering import StaticFeature


class OrderFrequency(StaticFeature):

	def __init__(self, data_store):
		super().__init__(data_store, 'order_frequency')

	def _compute_feature(self):
		# temporarily set to NaN for accurate computation, since it is set to -1 by superclass beforehand
		self.orders_tip.loc[self.orders_tip['order_number'] == 1, 'days_since_prior_order'] = np.nan
		self.orders_tip[self.feature] = (self.orders_tip.groupby('user_id')['days_since_prior_order'].expanding()
										 .mean().reset_index(level=0, drop=True))
		# set back to -1
		self.orders_tip.loc[self.orders_tip['order_number'] == 1, 'days_since_prior_order'] = -1
		self.orders_tip.loc[self.orders_tip['order_number'] == 1, self.feature] = -1

	def _analyze_feature(self):
		pass
