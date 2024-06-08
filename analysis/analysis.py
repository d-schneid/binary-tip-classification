from abc import ABC, abstractmethod


class Analysis(ABC):

    def __init__(self, data_manager):
        self.orders_tip = data_manager.get_orders_tip(complete=True)
        self.orders_tip = self.orders_tip[self.orders_tip['eval_set'] == 'prior']
        self.orders_joined_complete = data_manager.get_orders_joined(complete=True)
        self.orders_joined = self.orders_joined_complete[self.orders_joined_complete['eval_set'] == 'prior']
        self.products = data_manager.get_products()
        self.departments = data_manager.get_departments()
        self.aisles = data_manager.get_aisles()
        self.save_path = 'data/plots/'

    def execute_analysis(self, save_plots=False):
        self._analyze()
        self._show_results(save_plots)

    def _save_plot(self, fig, file):
        fig.savefig(self.save_path + file, dpi=1000)

    @abstractmethod
    def _analyze(self):
        pass

    @abstractmethod
    def _show_results(self, save_plots=False):
        pass
