import numpy as np
import pandas as pd

from feature_engineering import StaticFeature


class ModeDepartment(StaticFeature):

	def __init__(self, data_store):
		super().__init__(data_store, 'mode_dept')

	def _compute_feature(self):
		mode_dept = (self.orders_joined[['order_id', 'department_id']].groupby('order_id')['department_id']
					 .agg(self._compute_mode).reset_index())
		mode_dept = mode_dept.rename(columns={'department_id': self.feature})
		self.orders_tip = pd.merge(self.orders_tip, mode_dept, on='order_id', how='left')

	def _compute_mode(self, dept_ids):
		modes = pd.Series.mode(dept_ids)
		if len(modes) == len(dept_ids):
			return -1  # All values are unique, no mode
		elif len(modes) > 1:
			return np.random.choice(modes)  # Multiple modes, choose randomly
		else:
			return modes[0]

	def _analyze_feature(self):
		pass
