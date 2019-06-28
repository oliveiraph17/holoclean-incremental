import math

from tqdm import tqdm

from ..estimator import Estimator
from ..domain import DomainEngine
from utils import NULL_REPR


class NaiveBayes(Estimator):
    """
    NaiveBayes is an estimator of posterior probabilities using the naive independence assumption
        p(v_cur | v_init) = p(v_cur) * product_i (v_init_i | v_cur),
    where 'v_init_i' is the initial value corresponding to attribute 'i'.
    This probability is normalized over all values passed to predict_pp.
    """
    def __init__(self, env, dataset, domain_df, correlations):
        Estimator.__init__(self, env, dataset)

        self.raw_tuples, self.new_tuples, self.single_freq, self.pair_freq, _, _ = self.ds.get_statistics()
        self.domain_df = domain_df
        self._correlations = correlations
        self._cor_strength = self.env['nb_cor_strength']
        self._corr_attrs = {}

        # TID to raw data tuple for prediction.
        self._raw_records_by_tid = {}
        for row in self.ds.get_raw_data().to_records():
            self._raw_records_by_tid[row['_tid_']] = row

    def train(self):
        pass

    def predict_pp(self, row, attr, values):
        nb_score = []
        correlated_attributes = DomainEngine.get_corr_attributes(attr,
                                                                 self._cor_strength,
                                                                 self._correlations,
                                                                 self._corr_attrs)

        for val1 in values:
            val1_count = self.single_freq[attr][val1]
            log_prob = math.log(float(val1_count) / float(self.raw_tuples))
            for at in correlated_attributes:
                # Ignore same attribute, index, and tuple id.
                if at == attr or at == '_tid_':
                    continue
                val2 = row[at]
                # Since we do not have co-occurrence stats with NULL values,
                # we skip them.
                # It also doesn't make sense for our likelihood to be conditioned
                # on a NULL value.
                if val2 == NULL_REPR:
                    continue
                val2_val1_count = 0.1
                if val1 in self.pair_freq[attr][at]:
                    if val2 in self.pair_freq[attr][at][val1]:
                        val2_val1_count = max(self.pair_freq[attr][at][val1][val2] - 1.0, 0.1)
                p = float(val2_val1_count) / float(val1_count)
                log_prob += math.log(p)
            nb_score.append((val1, log_prob))

        denom = sum(map(math.exp, [log_prob for _, log_prob in nb_score]))

        for val, log_prob in nb_score:
            yield (val, math.exp(log_prob) / denom)

    def predict_pp_batch(self):
        """
        Performs batch prediction.

        This technically invokes predict_pp underneath.

        Returns a List[List[Tuple]] where each List[Tuple] corresponds to
        a cell (ordered by the order a cell appears in `self.domain_df`
        during construction) and each Tuple is (val, proba) where
        val is the domain value and proba is the estimator's posterior probability estimate.
        """
        for row in tqdm(self.domain_df.to_records()):
            yield self.predict_pp(self._raw_records_by_tid[row['_tid_']], row['attribute'], row['domain'].split('|||'))
