from abc import ABC, abstractmethod


class Analysis(ABC):
    
    def __init__(self, data_manager):
        self.orders_tip = data_manager.get_orders_tip(complete=True)
        self.orders_tip = self.orders_tip[self.orders_tip['eval_set'] == 'prior']
        self.orders_joined = data_manager.get_orders_joined()
        self.products = data_manager.get_products()
        self.departments = data_manager.get_departments()
        self.aisles = data_manager.get_aisles()

    def execute_analysis(self):
        self._analyze()
        self._show_results()

    @abstractmethod
    def _analyze(self):
        pass

    @abstractmethod
    def _show_results(self):
        pass
