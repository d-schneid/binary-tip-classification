from abc import ABC, abstractmethod


class Feature(ABC):

	def __init__(self, data_store, name):
		self.data_store = data_store
		self.feature = name
		self.orders_tip = self._get_orders_tip()
		self.orders_joined = self._get_orders_joined()

	@abstractmethod
	def _get_orders_tip(self):
		pass

	@abstractmethod
	def _get_orders_joined(self):
		pass

	def compute_feature(self):
		self._handle_missing_values()
		self._compute_feature()
		if self._reference_outdated():
			self._update_data_store()
			self._refresh_references()

	def _handle_missing_values(self):
		self.orders_tip['days_since_prior_order'] = self.orders_tip['days_since_prior_order'].fillna(-1).astype(int)
		pass

	@abstractmethod
	def _compute_feature(self):
		pass

	def analyze_feature(self):
		pass

	@abstractmethod
	def _analyze_feature(self):
		pass

	def _refresh_references(self):
		self.orders_tip = self._get_orders_tip()
		self.orders_joined = self._get_orders_joined()

	@abstractmethod
	def _update_data_store(self):
		pass

	def _reference_outdated(self):
		return self.orders_tip is not self._get_orders_tip()


class StaticFeature(Feature):

	def __init__(self, data_store, name):
		super().__init__(data_store, name)

	def _get_orders_tip(self):
		return self.data_store.get_orders_tip()

	def _get_orders_joined(self):
		return self.data_store.get_orders_joined()

	def compute_feature(self):
		self._refresh_references()
		if self.feature not in self.orders_tip.columns:
			super().compute_feature()

	@abstractmethod
	def _compute_feature(self):
		pass

	@abstractmethod
	def _analyze_feature(self):
		pass

	def _update_data_store(self):
		self.data_store.merge_orders_tip(self.orders_tip, self.feature)


class DynamicFeature(Feature):

	def __init__(self, data_store, name):
		super().__init__(data_store, name)

	def _get_orders_tip(self):
		return self.data_store.get_orders_tip_subset()

	def _get_orders_joined(self):
		return self.data_store.get_orders_joined_subset()

	def compute_feature(self):
		self._refresh_references()
		super().compute_feature()

	@abstractmethod
	def _compute_feature(self):
		pass

	@abstractmethod
	def _analyze_feature(self):
		pass

	def _update_data_store(self):
		self.data_store.merge_orders_tip_subset(self.orders_tip, self.feature)
