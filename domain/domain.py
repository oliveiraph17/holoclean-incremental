import logging
import pandas as pd
import time

import itertools
import numpy as np
from tqdm import tqdm

from dataset import AuxTables, CellStatus
from .estimators import NaiveBayes
from utils import NULL_REPR


class DomainEngine:
    def __init__(self, env, dataset, max_sample=5):
        """
        :param env: (dict) contains global settings such as verbose.
        :param dataset: (Dataset) current dataset.
        :param max_sample: (int) maximum # of domain values from a random sample.
        """
        self.env = env
        self.ds = dataset
        self.domain_thresh_1 = env["domain_thresh_1"]
        self.weak_label_thresh = env["weak_label_thresh"]
        self.domain_thresh_2 = env["domain_thresh_2"]
        self.max_domain = env["max_domain"]
        self.setup_complete = False
        self.active_attributes = None
        self.domain = None
        self.total = 0
        self.correlations = None
        self.corr_attrs = {}
        self.cor_strength = env["cor_strength"]
        self.max_sample = max_sample
        self.single_stats = {}
        self.pair_stats = {}
        self.all_attrs = {}

    def setup(self):
        """
        Initializes the in-memory and PostgreSQL auxiliary tables (e.g. 'cell_domain', 'pos_values').
        """
        tic = time.time()
        self.setup_attributes()
        domain = self.generate_domain()
        self.store_domains(domain)
        status = "DONE with domain preparation"
        toc = time.time()
        return status, toc - tic

    def store_domains(self, domain):
        """
        Generates the domain DataFrame as the 'cell_domain' auxiliary table,
        as well as generates the 'pos_values' auxiliary table,
        and stores them in PostgreSQL.

        Schema for cell_domain:
            _cid_: cell ID
            _tid_: entity/tuple ID
            _vid_: random variable ID (all cells with more than 1 domain value)
            attribute: column name
            domain: top co-occurring values (within correlation threshold) of 'attribute'
            domain_size: number of values in 'domain'
            fixed: 1 if a random sample was taken in the absence of correlated attributes or top co-occurring values,
                   0 otherwise
            init_index: domain index of 'init_value'
            init_value: initial value for this cell
            weak_label: value assigned as ground-truth to this cell
            weak_label_idx: domain index of 'weak_label'

        Schema for pos_values:
            _vid_: random variable ID (all cells with more than 1 domain value)
            _tid_: entity/tuple ID
            _cid_: cell ID
            attribute: column name
            rv_val: a value from the domain of the current random variable
            val_id: a numeric identifier for rv_val starting from 1
        """
        if domain.empty:
            raise Exception("ERROR: Generated domain is empty.")
        else:
            self.ds.generate_aux_table(AuxTables.cell_domain, domain, store=True, index_attrs=['_vid_'])
            self.ds.aux_table[AuxTables.cell_domain].create_db_index(self.ds.engine, ['_tid_'])
            self.ds.aux_table[AuxTables.cell_domain].create_db_index(self.ds.engine, ['_cid_'])

            query = "SELECT _vid_, _cid_, _tid_, attribute, a.rv_val, a.val_id FROM %s" % AuxTables.cell_domain.name
            query += ", unnest(string_to_array(regexp_replace(domain, \'[{\"\"}]\', \'\', \'gi\'), \'|||\'))"
            query += "WITH ORDINALITY a(rv_val, val_id)"

            self.ds.generate_aux_table_sql(AuxTables.pos_values, query, index_attrs=['_tid_', 'attribute'])

    def setup_attributes(self):
        self.active_attributes = self.get_active_attributes()
        total, single_stats, pair_stats = self.ds.get_statistics()
        self.correlations = self.ds.get_correlations()

        # PH: the class variable 'total' is never used.
        self.total = total
        self.single_stats = single_stats

        logging.debug("Preparing pruned co-occurring statistics...")
        tic = time.clock()
        self.pair_stats = self._pruned_pair_stats(pair_stats)
        toc = time.clock()
        logging.debug("DONE with pruned co-occurring statistics in %.2f secs", toc - tic)

        self.setup_complete = True

    def _pruned_pair_stats(self, pair_stats):
        """
        Converts 'pair_stats', which is a dictionary of the form
            { attr1 -> { attr2 -> { val1 -> { val2 -> count } } } }, where
              <val1>: all possible values for attr1,
              <val2>: all values for attr2 that appear at least once with <val1>, and
              <count>: frequency (# of entities) where attr1=<val1> AND attr2=<val2>,
        to a flattened 4-level dictionary of the form
            { attr1 -> { attr2 -> { val1 -> [pruned list of val2] } } }.

        It keeps only the co-occurring values of attr2 that exceed 'self.domain_thresh_1'.
        """

        out = {}
        for attr1 in tqdm(pair_stats.keys()):
            out[attr1] = {}
            for attr2 in pair_stats[attr1].keys():
                out[attr1][attr2] = {}
                for val1 in pair_stats[attr1][attr2].keys():
                    denominator = self.single_stats[attr1][val1]

                    # 'tau' becomes a threshold on co-occurrence frequency
                    # based on the co-occurrence probability threshold 'self.domain_thresh_1'.
                    tau = float(self.domain_thresh_1 * denominator)

                    top_cands = [val2 for (val2, count) in pair_stats[attr1][attr2][val1].items() if count > tau]
                    out[attr1][attr2][val1] = top_cands
        return out

    def get_active_attributes(self):
        """
        Returns the attributes to be modeled, which are the ones having at least one potentially erroneous cell.
        """
        query = 'SELECT DISTINCT attribute as attribute FROM {}'.format(AuxTables.dk_cells.name)
        result = self.ds.engine.execute_query(query)

        if not result:
            raise Exception("No attribute contains erroneous cells.")

        # Sort the active attributes to maintain the order of the random variable IDs.
        return sorted(itertools.chain(*result))

    @staticmethod
    def get_corr_attributes(attr, thres, corr_lst, corr_attrs_lst):
        """
        Returns attributes from 'correlations_lst' that are correlated with 'attr'
        within a magnitude of 'thres'.

        :param attr: (string) the original attribute to get the correlated attributes for.
        :param thres: (float) correlation threshold for attributes to be returned.
        :param corr_lst: (dict) correlations between every pair of attributes.
        :param corr_attrs_lst: (list[list[(attr, thres)]]) correlated attributes to 'attr' within 'thres'.
               The list is populated in this method and the sublist corresponding to 'attr' is returned.

        :return (list) attributes correlated to 'attr' within the threshold 'thres'.
        """
        # Not memoized: find correlated attributes from correlation dictionary.
        if (attr, thres) not in corr_attrs_lst:
            corr_attrs_lst[(attr, thres)] = []

            if attr in corr_lst:
                attr_correlations = corr_lst[attr]
                corr_attrs_lst[(attr, thres)] = sorted([corr_attr
                                                        for corr_attr, corr_strength in attr_correlations.items()
                                                        if corr_attr != attr and corr_strength > thres])

        return corr_attrs_lst[(attr, thres)]

    def generate_domain(self):
        """
        Generates the domain for each cell in the active attributes,
        as well as assigns a random variable ID ('_vid_') to cells with at least 2 domain values.

        See 'get_domain_cell' for how the domain is generated from co-occurrences and correlated attributes.

        If no values can be found from correlated attributes, then return a random sample of domain values.

        :return: DataFrame with columns
            _tid_: entity/tuple ID
            _cid_: cell ID (one for every data cell in the active attributes)
            _vid_: random variable ID (one for every cell with at least 2 domain values)
            attribute: attribute name
            domain: string of domain values separated by '|||'
            domain_size: number of domain values
            init_value: initial value for this cell
            init_value_idx: domain index of init_value
            fixed: 2 if a random sample was taken in the absence of correlated attributes or top co-occurring values
        """

        if not self.setup_complete:
            raise Exception("Perform error detection and call <setup_attributes> first.")

        logging.debug('Generating initial set of unpruned domain values...')

        tic = time.clock()

        cells = []
        vid = 0

        if self.ds.is_first_batch() or not self.env['repair_previous_errors']:
            records = self.ds.get_raw_data().to_records(index=False)
        else:
            records = pd.concat([self.ds.get_previous_dirty_rows(), self.ds.get_raw_data()]).to_records(index=False)

        self.all_attrs = list(records.dtype.names)

        for row in tqdm(list(records)):
            tid = row['_tid_']

            for attr in self.active_attributes:
                init_value, init_value_idx, dom = self.get_domain_cell(attr, row)
                # We use an estimator model for additional weak labelling, which requires an initial pruned domain.
                # Weak labels are trained on the initial values.
                cid = self.ds.get_cell_id(tid, attr)

                # Originally, all cells have a NOT_SET status to be considered in weak labelling.
                cell_status = CellStatus.NOT_SET.value

                if len(dom) <= 1:
                    # We could not come up with a domain and the initial value is NULL.
                    # Therefore, a random domain is not likely to help us.
                    # We ignore this cell and continue.
                    if init_value == NULL_REPR:
                        continue

                    # Not enough domain values, so we need some random values (other than 'init_value') for training.
                    # However, this might still get us zero additional values.
                    rand_dom_values = self.get_random_domain(attr, init_value)

                    # The set of additional domain values might still be empty.
                    # In this case, there are no other possible values for this cell.
                    # There is no point in using this cell for training nor in running inference on it,
                    # since we could not even generate a random domain.
                    # Therefore, we will not include it in the final tensor.
                    if len(rand_dom_values) == 0:
                        continue

                    # Otherwise, add the random additional values to the domain.
                    dom.extend(rand_dom_values)

                    # PH: I added this because some domains in the 'cell_domain' table were unsorted.
                    dom = sorted(dom)
                    init_value_idx = dom.index(init_value)

                    # There was originally just a single value for the domain of this cell.
                    # Other values were randomly assigned to the domain.
                    # Therefore, these will not be modified by the estimator.
                    cell_status = CellStatus.SINGLE_VALUE.value

                cells.append({"_tid_": tid,
                              "attribute": attr,
                              "_cid_": cid,
                              "_vid_": vid,
                              "domain": "|||".join(dom),
                              "domain_size": len(dom),
                              "init_value": init_value,
                              "init_index": init_value_idx,
                              "weak_label": init_value,
                              "weak_label_idx": init_value_idx,
                              "fixed": cell_status})

                vid += 1

        domain_df = pd.DataFrame(data=cells).sort_values('_vid_')
        logging.debug('DONE generating initial set of domain values in %.2f secs', time.clock() - tic)

        if self.env['weak_label_thresh'] == 1 and self.env['domain_thresh_2'] == 0:
            # Skip estimator if we require no weak labelling nor domain pruning based on posterior probabilities.
            return domain_df

        # Feed the Naive Bayes estimator with pruned domain values from correlated attributes.
        logging.debug('Training posterior model for estimating domain value probabilities...')
        tic = time.clock()
        estimator = NaiveBayes(self.env, self.ds, domain_df, self.correlations)
        logging.debug('DONE training posterior model in %.2f secs', time.clock() - tic)

        # Predict probabilities for all pruned domain values.
        logging.debug('Predicting domain value probabilities from posterior model...')
        tic = time.clock()
        preds_by_cell = estimator.predict_pp_batch()
        logging.debug('DONE predicting in %.2f secs', time.clock() - tic)

        logging.debug('Assembling final \'cell_domain\' table...')
        tic = time.clock()

        # Iterate through current dataset and generate posterior probabilities for weak labelling.
        num_weak_labels = 0
        updated_domain_df = []
        for preds, row in tqdm(zip(preds_by_cell, domain_df.to_records())):
            # Do not relabel single-valued cells.
            if row['fixed'] == CellStatus.SINGLE_VALUE.value:
                updated_domain_df.append(row)
                continue

            # Prune domain if any of the values are below the 'domain_thresh_2' parameter.
            preds = [[val, prob] for val, prob in preds if prob >= self.domain_thresh_2] or preds

            # Cap the maximum number of domain values to self.max_domain based on probabilities.
            domain_values = [val for val, prob in sorted(preds,
                                                         key=lambda pred: pred[1],
                                                         reverse=True)[:self.max_domain]]

            # Ensure the initial value is included even if its probability is low.
            if row['init_value'] not in domain_values and row['init_value'] != NULL_REPR:
                domain_values.append(row['init_value'])

            domain_values = sorted(domain_values)

            # Update our memoized domain values for this row.
            row['domain'] = '|||'.join(domain_values)
            row['domain_size'] = len(domain_values)

            # This check occurs because 'init_value' would not be in 'domain_values' if it were NULL.
            if row['init_value'] in domain_values:
                # Update 'init_index' based on new domain.
                row['init_index'] = domain_values.index(row['init_value'])

            # Update 'weak_label_idx' based on new domain.
            if row['weak_label'] != NULL_REPR:
                row['weak_label_idx'] = domain_values.index(row['weak_label'])

            weak_label, weak_label_prob = max(preds, key=lambda pred: pred[1])

            # Assign weak label if it is different than the initial value
            # and the domain value probability exceeds 'self.weak_label_thresh'.
            if weak_label != row['init_value'] and weak_label_prob >= self.weak_label_thresh:
                num_weak_labels += 1

                weak_label_idx = domain_values.index(weak_label)
                row['weak_label'] = weak_label
                row['weak_label_idx'] = weak_label_idx
                row['fixed'] = CellStatus.WEAK_LABEL.value

            updated_domain_df.append(row)

        # Update cell domain DataFrame with new updated domain.
        domain_df = pd.DataFrame.from_records(updated_domain_df, columns=updated_domain_df[0].dtype.names)\
            .drop('index', axis=1)\
            .sort_values('_vid_')

        logging.debug('DONE assembling \'cell_domain\' table in %.2f secs', time.clock() - tic)

        logging.info('Number of (additional) weak labels assigned from posterior model: %d', num_weak_labels)

        logging.debug('DONE generating domain and weak labels')
        return domain_df

    def get_domain_cell(self, attr, row):
        """
        Returns a list of all domain values for the given entity (row) and attribute.
        The domain never has NULL as a possible value.

        We define domain values as values in 'attr' co-occurring with values in attributes ('cond_attr')
        that are correlated with 'attr' within a magnitude of 'self.cor_strength' (provided as a parameter).

        For example:

                cond_attr  |  attr
                H             B    <- current row
                H             C
                I             D
                H             E

        This would produce [B, C, E] as domain values.

        :return: (initial value of entity-attribute, domain values for entity-attribute).
        """

        domain = set()
        init_value = row[attr]
        correlated_attributes = self.get_corr_attributes(attr, self.cor_strength, self.correlations, self.corr_attrs)

        # Iterate through all correlated attributes and take the top co-occurring values of 'attr'
        # with respect to the value of 'cond_attr' in the current row.
        for cond_attr in correlated_attributes:
            if cond_attr == attr or cond_attr == '_tid_':
                continue

            if not self.pair_stats[cond_attr][attr]:
                logging.warning("There are no pair statistics between attributes: {}, {}".format(cond_attr, attr))
                continue

            cond_val = row[cond_attr]

            # Ignore co-occurrence when 'cond_val' is NULL.
            # It does not make sense to retrieve the top co-occurring values with a NULL value.
            # Moreover, ignore co-occurrence when 'cond_val' only co-occurs with NULL values.
            if cond_val == NULL_REPR or self.pair_stats[cond_attr][attr][cond_val] == [NULL_REPR]:
                continue

            # Update domain with the top co-occurring values with 'cond_val'.
            candidates = self.pair_stats[cond_attr][attr][cond_val]
            domain.update(candidates)

        domain.discard(NULL_REPR)
        assert NULL_REPR not in domain

        # Add the initial value to the domain if it is not NULL.
        if init_value != NULL_REPR:
            domain.add(init_value)

        # Convert to ordered list to preserve order.
        domain_lst = sorted(list(domain))

        # Get the index of the initial value.
        # NULL values are not in the domain, so we set their index to -1.
        init_value_idx = -1
        if init_value != NULL_REPR:
            init_value_idx = domain_lst.index(init_value)

        return init_value, init_value_idx, domain_lst

    def get_random_domain(self, attr, cur_value):
        """
        Returns a random sample of at most size 'self.max_sample'
        containing domain values of 'attr' that are different than 'cur_value'.
        """
        domain_pool = set(self.single_stats[attr].keys())

        domain_pool.discard(NULL_REPR)
        assert NULL_REPR not in domain_pool

        domain_pool.discard(cur_value)
        domain_pool = sorted(list(domain_pool))

        size = len(domain_pool)

        if size > 0:
            k = min(self.max_sample, size)
            additional_values = np.random.choice(domain_pool, size=k, replace=False)
        else:
            additional_values = []

        return sorted(additional_values)
