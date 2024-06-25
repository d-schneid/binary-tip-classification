import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import groupby

from analysis import Analysis


class TipSequence(Analysis):

	def __init__(self, data_manager):
		super().__init__(data_manager)
		self._user_mean_tip_streaks = None
		self._mean_tip_streak_per_order_num = None

	def _analyze(self):
		# first analysis
		user_mean_tip_streaks = (self.orders_tip[['user_id', 'tip']].groupby('user_id').
								 apply(self._compute_user_mean_tip_streak, include_groups=False).reset_index())
		user_mean_tip_streaks.columns = ['user_id', 'mean_tip_streak']

		user_num_orders = self.orders_tip.groupby('user_id')[['order_number']].max().reset_index()
		user_num_orders.columns = ['user_id', 'num_orders']
		user_total_tips = self.orders_tip.groupby('user_id')['tip'].sum().reset_index()
		user_total_tips.columns = ['user_id', 'num_tips']
		user_var_tip_streaks = (self.orders_tip.groupby('user_id').
								apply(self._compute_user_var_tip_streak, include_groups=False).reset_index())
		user_var_tip_streaks.columns = ['user_id', 'var_tip_streak']

		user_mean_tip_streaks = pd.merge(user_mean_tip_streaks, user_num_orders)
		user_mean_tip_streaks = pd.merge(user_mean_tip_streaks, user_total_tips)
		user_mean_tip_streaks = pd.merge(user_mean_tip_streaks, user_var_tip_streaks)
		user_mean_tip_streaks['streaks_per_order'] = user_mean_tip_streaks['mean_tip_streak'] / user_mean_tip_streaks['num_orders']
		user_mean_tip_streaks = user_mean_tip_streaks.infer_objects(copy=False).fillna(-1)
		self._user_mean_tip_streaks = user_mean_tip_streaks

		# second analysis
		df_preprocess = self.orders_tip.copy()
		df_preprocess['tip'] = df_preprocess['tip'].astype(int)
		df_preprocess['sequence_diff'] = df_preprocess['tip'].diff()
		df_preprocess['new_sequence'] = ((df_preprocess['sequence_diff'] != 0) &
														 (df_preprocess['tip'] == 1))
		df_preprocess['tip_sequence'] = df_preprocess['new_sequence'].astype(int)
		df_preprocess['cumulative_tip_sequences'] = df_preprocess.groupby('user_id')['tip_sequence'].cumsum()
		df_preprocess['cumulative_sum_of_tips'] = df_preprocess[df_preprocess['tip'] == 1].groupby('user_id')['tip'].cumsum()
		df_preprocess['cumulative_sum_of_tips'] = df_preprocess.groupby('user_id')['cumulative_sum_of_tips'].ffill().fillna(0)
		df_preprocess = df_preprocess.drop(columns=['sequence_diff', 'new_sequence', 'tip_sequence'])
		df_preprocess['ratio'] = np.where(df_preprocess['cumulative_tip_sequences'] != 0,
										  df_preprocess['cumulative_sum_of_tips'] / df_preprocess['cumulative_tip_sequences'], 0)

		self._mean_tip_streak_per_order_num = df_preprocess.groupby('order_number')['ratio'].agg(['mean', 'var']).reset_index()
		self._mean_tip_streak_per_order_num = (self._mean_tip_streak_per_order_num.
											   rename(columns={'mean': 'mean_tip_streak_users', 'var': 'var_tip_streak_users'}))

	def _compute_user_mean_tip_streak(self, user_df):
		user_df["streak_id"] = (user_df["tip"] == 0).cumsum()
		streak_len = user_df[user_df["tip"] == 1].groupby("streak_id").size()
		return streak_len.mean()

	def _compute_user_var_tip_streak(self, user_df):
		user_df = user_df.sort_values('order_number')
		tip_streaks = [sum(1 for _ in group) for key, group in groupby(user_df['tip']) if key == 1]
		return np.var(tip_streaks) if tip_streaks else -1

	def _show_results(self, save_plots=False):
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

		ax1.scatter(self._user_mean_tip_streaks['num_orders'], self._user_mean_tip_streaks['mean_tip_streak'])
		ax1.set_xlabel('Total number of orders')
		ax1.set_ylabel('Mean tip streak')
		ax1.set_title('Total Number of Orders for each User and the corresponding Mean Tip Streak Across all respective Orders')

		ax2.hist(self._user_mean_tip_streaks['var_tip_streak'], bins='auto', edgecolor='black')
		ax2.set_xlabel('Variance of tip streak')
		ax2.set_xlim([-1, 20])
		ax2.set_ylabel('Frequency')
		ax2.set_title('Histogram of Tip Streak Variance of Users across all respective Orders')

		#ax3.scatter(self._mean_tip_streak_per_order_num['order_number'],
		#			self._mean_tip_streak_per_order_num['mean_tip_streak_users'], c='blue', label='Mean tip streak')
		#ax3.scatter(self._mean_tip_streak_per_order_num['order_number'],
		#			self._mean_tip_streak_per_order_num['var_tip_streak_users'], c='red', label='Variance of tip streak')
		#ax3.set_title('Mean/Variance of Tip Streaks up to each Order Number across all Users')
		#ax3.set_xlabel('Order number')
		#ax3.set_ylabel('Value')
		#ax3.legend(loc='upper left')

		plt.tight_layout()
		plt.show()

		if save_plots:
			self._save_plot(fig, 'tip_sequence.png')
