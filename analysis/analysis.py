from abc import ABC, abstractmethod


class Analysis(ABC):

    def __init__(self, data_manager):
        self.orders_tip = data_manager.get_orders_tip(complete=True)
        self.orders_joined = data_manager.get_orders_joined()
        self.products = data_manager.get_products()
        self.departments = data_manager.get_departments()
        self.aisles = data_manager.get_aisles()

    @abstractmethod
    def analyze(self):
        pass

    @abstractmethod
    def show_results(self):
        pass
