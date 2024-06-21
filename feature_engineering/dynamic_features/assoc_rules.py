from abc import abstractmethod
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

from feature_engineering import DynamicFeature


class AssocRules(DynamicFeature):

    def __init__(self, name, id_col, min_support=0.001, min_confidence=0.0):
        super().__init__(name)
        self._id_col = id_col
        self._min_support = min_support
        self._min_confidence = min_confidence
        self._tip_indicator = -1
        self._assoc_rules = None
        self.feature_type = self.STEADY_FEATURE

    def _compute_feature(self):
        transactions = (self.orders_joined[['order_id', 'tip', self._id_col]].groupby('order_id').
                        apply(self._build_transaction).tolist())
        te = TransactionEncoder()
        transactions_df = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)

        # frequent itemsets contain transactions ids
        freq_itemsets = fpgrowth(transactions_df, min_support=self._min_support, use_colnames=True)
        assoc_rules = association_rules(freq_itemsets, metric="confidence", min_threshold=self._min_confidence)
        # use association rules as classifier for tip prediction
        self._assoc_rules = assoc_rules[assoc_rules['consequents'] == frozenset([self._tip_indicator])]

        # binary representation of the content (wrt 'self._id_col') of each order
        orders_bin = (self.orders_joined[['order_id', self._id_col]].
                      pivot_table(index='order_id', columns=self._id_col, aggfunc=len, fill_value=0).gt(0).reset_index())
        orders_tip_temp = self.orders_tip[['order_id']].copy()
        # order_id of the first encountered order of duplicates is used as a representative of the equivalence class
        # of duplicates
        # assign each order the representative of its corresponding equivalence class
        orders_tip_temp['fst_dup_id'] = orders_bin.groupby(orders_bin.columns.difference(['order_id']).
                                                           tolist())['order_id'].transform('first')
        # drop duplicates for an efficient computation
        unique_orders_bin = orders_bin.drop_duplicates(subset=orders_bin.columns.difference(['order_id']), keep='first')

        # compute number of overlapping ids between each order and each right-hand side of an association rule for
        # finding the association rule that best represents an order (= maximum subset)
        subset_overlap = unique_orders_bin.set_index('order_id').dot(self._get_assoc_rules_bin().transpose())
        max_overlap_index = subset_overlap.idxmax(axis=1)

        # retrieve confidence (i.e. tip rate) of matching association rule for each representative of a duplicate
        # equivalence class
        tip_rate_fst_dups = max_overlap_index.map(self._assoc_rules['confidence'])
        tip_rate_fst_dups.name = 'tip_rate_fst_dups'
        orders_tip_temp = orders_tip_temp.merge(tip_rate_fst_dups.to_frame(), on='order_id', how='left')
        # assign each order of the same duplicate equivalance class the retrieved confidence (i.e. tip rate)
        orders_tip_temp[self.feature] = orders_tip_temp.groupby('fst_dup_id')[tip_rate_fst_dups.name].ffill().bfill()

        self.orders_tip = pd.merge(self.orders_tip, orders_tip_temp[['order_id', self.feature]], on='order_id', how='left')

        if self.orders_tip[self.feature].isna().any():
            raise ValueError("Not every order contains a tip rate based on association rules!")

    def _build_transaction(self, order):
        transaction = order[self._id_col].unique().tolist()
        if order['tip'].iloc[0]:
            transaction.append(self._tip_indicator)
        return transaction

    def _get_assoc_rules_bin(self):
        assoc_rules_copy = self._assoc_rules[['antecedents', 'confidence']].copy()
        # Get the unique ids in `antecedents`
        ids = set(id_ for rule in assoc_rules_copy['antecedents'] for id_ in rule)
        # Map each id to a new column
        for id_ in ids:
            assoc_rules_copy[id_] = assoc_rules_copy['antecedents'].apply(lambda rule: int(id_ in rule))
        return assoc_rules_copy.drop(['antecedents', 'confidence'], axis=1)


class AssocRulesDepartments(AssocRules):

    def __init__(self, min_support=0.001, min_confidence=0.0):
        super().__init__('tip_rate_assoc_rules_depts', 'department_id', min_support, min_confidence)


class AssocRulesAisles(AssocRules):

    def __init__(self, min_support=0.001, min_confidence=0.0):
        super().__init__('tip_rate_assoc_rules_aisles', 'aisle_id', min_support, min_confidence)
