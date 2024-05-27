from feature_engineering import StaticFeature
from scipy.stats import pearsonr


class SimOrdersTipRatio(StaticFeature):

    def __init__(self, data_store, similarity_percentage):
        super().__init__(data_store, 'sim_orders_tip_ratio')
        self.similarity_percentage = similarity_percentage

    def _compute_feature(self):

        # Ähnlichkeitsfaktor Wie viele Produkte stimmen überein? Angabe in Prozent
        # Range mit ähnlichen allen orders wie ähnlich sind die und dann --> Prozentwert festlegen

        orders_tip = self.orders_tip.copy()
        orders_tip[self.feature] = orders_tip.apply(self._calculate_similarity_tip_ratio, axis=1)

    def _calculate_similarity_tip_ratio(self, order):
        tip_counter = 0
        similarity_counter = 0

        products_in_current_order = self.orders_joined[self.orders_joined['order_id'] == order.order_id][
            'product_id'].toList()
        for i in range(order['order_number'] - 1, 0, -1):
            products_in_order = self.orders_joined[
                (self.orders_joined['user_id'] == order.user_id) & self.orders_joined['order_number'] == i][
                'product_id'].toList()
            correlation = pearsonr(products_in_current_order, products_in_order)
            if correlation[1] >= self.similarity_percentage:
                similarity_counter += 1
                # if tip of order == true --> tip_counter + 1

        return tip_counter / similarity_counter

    def _analyze_feature(self):
        pass
