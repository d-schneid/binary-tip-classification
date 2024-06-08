import pandas as pd
from matplotlib import pyplot as plt

from analysis import Analysis


class GeneralAnalysis(Analysis):

    def __init__(self, data_manager):
        super().__init__(data_manager)

    def _analyze(self):
        pass

    def _show_results(self, save_plots=False):
        data = self.orders_joined_complete

        # General Facts
        print(f"General facts about the dataset:")
        print(data.info())
        print(data.shape)
        print("\n")

        # Data Statistics
        print("Size of the Dataset:")
        print(f"Number of prior Data {data[data['eval_set'] == 'prior'].count()['order_id']}")
        print(f"Number of train Data {data[data['eval_set'] == 'train'].count()['order_id']}")
        print("\n")

        print("General Information about the object data:")
        print(data.describe(include='object'))
        print("\n")

        print("General Information about the numerical data:")
        print(data.describe())
        print("\n")

        # Existence of NaN values?
        print("Existence of NaN values in complete dataset:")
        print(data.isnull().sum())
        print("Existence of NaN values in prior dataset:")
        print(data[data['eval_set'] == "prior"].isnull().sum())
        print("\n")

