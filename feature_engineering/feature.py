from abc import ABC, abstractmethod


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

    def analyze_feature(self):
        pass

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
