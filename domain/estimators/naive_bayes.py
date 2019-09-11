import math
import pandas as pd

from tqdm import tqdm

from ..estimator import Estimator
from utils import NULL_REPR

class NaiveBayes(Estimator):
    """
    NaiveBayes is an estimator of posterior probabilities using the naive
    independence assumption where
        p(v_cur | v_init) = p(v_cur) * \prod_i (v_init_i | v_cur)
    where v_init_i is the init value for corresponding to attribute i. This
    probability is normalized over all values passed into predict_pp.
    """
    def __init__(self, env, dataset, domain_df, correlations):
        Estimator.__init__(self, env, dataset, domain_df)

        self._n_tuples, self._freq, self._cooccur_freq = self.ds.get_statistics()
        self._correlations = correlations
        self._cor_strength = self.env['nb_cor_strength']
        self._corr_attrs = {}

        # TID to raw data tuple for prediction.
        self._raw_records_by_tid = {}
        raw_df = self.ds.get_quantized_data() if self.ds.do_quantization else self.ds.get_raw_data()

        if not self.ds.is_first_batch() and self.env['repair_previous_errors']:
            # TODO(kaster): adjust here to properly handle quantized_data for previous dirty rows
            records = pd.concat([self.ds.get_previous_dirty_rows(), raw_df]).to_records()
        else:
            records = raw_df.to_records()

        for row in records:
            self._raw_records_by_tid[row['_tid_']] = row

    def train(self, num_epochs=None, batch_size=None):
        pass

    def _predict_pp(self, row, attr, values):
        from ..domain import DomainEngine
        nb_score = []
# <<<<<<< dev
#         correlated_attributes = self._get_corr_attributes(attr)
# =======
        correlated_attributes = DomainEngine.get_corr_attributes(attr,
                                                                 self._cor_strength,
                                                                 self._correlations)
# >>>>>>> dev-mixed
        for val1 in values:
            # This check was added recently, whereas the same check for 'val2' was already present.
            if val1 == NULL_REPR:
                continue

            val1_count = self._freq[attr][val1]
            log_prob = math.log(float(val1_count) / float(self._n_tuples))
            for at in correlated_attributes:
                # Ignore same attribute, index, and tuple id.
                if at == attr or at == '_tid_':
                    continue
                val2 = row[at]
                # It does not make sense for our likelihood to be conditioned
                # on a NULL value.
                if val2 == NULL_REPR:
                    continue
                val2_val1_count = 0.1
                if val1 in self._cooccur_freq[attr][at]:
                    if val2 in self._cooccur_freq[attr][at][val1]:
                        val2_val1_count = max(self._cooccur_freq[attr][at][val1][val2] - 1.0, 0.1)
                p = float(val2_val1_count) / float(val1_count)
                log_prob += math.log(p)
            nb_score.append((val1, log_prob))

        denom = sum(map(math.exp, [log_prob for _, log_prob in nb_score]))

        def val_probas():
            for val, log_prob in nb_score:
                yield val, math.exp(log_prob) / denom

        return val_probas()

    def predict_pp_batch(self):
        """
        Performs batch prediction.

        Returns (vid, True, List[Tuple]) where each List[Tuple] corresponds to
        a cell (ordered by the order a cell appears in self.domain_df
        during construction) and each Tuple is (val, proba) where
        val is the domain value and proba is the estimator's posterior probability estimate.
        """
        for row in tqdm(self.domain_df.to_records()):
            yield row['_vid_'], True, self._predict_pp(self._raw_records_by_tid[row['_tid_']], row['attribute'], row['domain'].split('|||'))
