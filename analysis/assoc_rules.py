import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt

from analysis import Analysis


class AssocRules(Analysis):

	def __init__(self, data_manager, id_col='department_id', min_support=0.001, min_confidence=0.0):
		super().__init__(data_manager)
		self._id_col = id_col
		self._min_support = min_support
		self._min_confidence = min_confidence
		self._tip_indicator = -1
		self._assoc_rules = None

	def get_assoc_rules(self):
		return self._assoc_rules

	def _analyze(self):
		transactions = (self.orders_joined[['order_id', 'tip', self._id_col]].groupby('order_id').
				  apply(self._build_transaction).tolist())
		te = TransactionEncoder()
		transactions_df = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)

		# frequent itemsets contain transactions ids
		freq_itemsets = fpgrowth(transactions_df, min_support=self._min_support, use_colnames=True)
		assoc_rules = association_rules(freq_itemsets, metric="confidence", min_threshold=self._min_confidence)
		# use association rules as classifier for tip prediction
		self._assoc_rules = assoc_rules[assoc_rules['consequents'] == frozenset([self._tip_indicator])]

	def _build_transaction(self, order):
		transaction = order[self._id_col].unique().tolist()
		if order['tip'].iloc[0]:
			transaction.append(self._tip_indicator)
		return transaction

	def _show_results(self, save_plots=False):
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

		ax1.scatter(self._assoc_rules['support'], self._assoc_rules['confidence'])
		ax1.set_xlabel('Support')
		ax1.set_ylabel('Confidence')
		ax1.set_title('Support vs Confidence')

		ax2.hist(self._assoc_rules['confidence'], bins='auto', edgecolor='black')
		ax2.set_xlabel('Confidence')
		ax2.set_ylabel('Frequency')
		ax2.set_title('Histogram of Confidence')
		ax2.set_xlim(0, 1)
		ax2.set_xticks(np.arange(0, 1.1, 0.1))

		plt.tight_layout()
		plt.show()

		if save_plots:
			self._save_plot(fig, 'assoc_rules.png')
