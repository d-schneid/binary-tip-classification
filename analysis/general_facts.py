import pandas as pd
from matplotlib import pyplot as plt

from analysis import Analysis


class GeneralFacts(Analysis):

    def __init__(self, data_manager):
        super().__init__(data_manager)

    def _analyze(self):
        pass

    def _show_results(self, save_plots=False):
        print(f"General facts about the dataset")
        print(f"Order specific analysis:")
        print(f"Total amount of orders        : {len(self.orders_tip)}")
        print(f"Total amount of users         : {self.orders_tip['user_id'].nunique()}")
        print(f"Average orders / user         : {self.orders_tip.groupby("user_id").size().mean()}")
        print(f"Average tipped orders / user  : {self.orders_tip.groupby(["user_id", "tip"], observed=True).size().mean()}")
        print(f"Overall order tip probability : {self.orders_tip['tip'].mean()}")

        print(f"----------\n")
        print(f"Product specific analysis:")
        print(f"Total amount of products       : {self.orders_joined['product_id'].nunique()}")
        print(f"Average products / order       : {self.orders_joined.groupby('order_id')['product_id'].size().mean()}")
        print(f"Average order amount / product : {self.orders_joined['product_id'].value_counts().mean()}")

        print(f"----------\n")
        print(f"Department specific analysis:")
        print(f"Total amount of departments       : {self.orders_joined['department_id'].nunique()}")
        print(f"Average departments / order       : {self.orders_joined.groupby('order_id')['department_id'].nunique().mean()}")
        print(f"Average products / department     : {self.orders_joined.groupby('department_id')['product_id'].nunique().mean()}")
        print(f"Average order amount / department : {self.orders_joined['department_id'].value_counts().mean()}")

        print(f"----------\n")
        print(f"Aisle specific analysis:")
        print(f"Total amount of aisles       : {self.orders_joined['aisle_id'].nunique()}")
        print(f"Average aisles / order       : {self.orders_joined.groupby('order_id')['aisle_id'].nunique().mean()}")
        print(f"Average products / aisle     : {self.orders_joined.groupby('aisle_id')['product_id'].nunique().mean()}")
        print(f"Average order amount / aisle : {self.orders_joined['aisle_id'].value_counts().mean()}")

