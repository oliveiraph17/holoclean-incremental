from enum import Enum
import logging
import os
import time

import pandas as pd
import numpy as np

from .dbengine import DBengine
from .table import Table, Source
from utils import dictify_df, NULL_REPR
from collections import namedtuple


class AuxTables(Enum):
    c_cells = 1
    dk_cells = 2
    cell_domain = 3
    pos_values = 4
    cell_distr = 5
    inf_values_idx = 6
    inf_values_dom = 7
    single_attr_stats = 8
    pair_attr_stats = 9


class CellStatus(Enum):
    NOT_SET = 0
    WEAK_LABEL = 1
    SINGLE_VALUE = 2


class Dataset:
    """
    This class keeps all DataFrames and tables for a HoloClean session.
    """
    def __init__(self, name, env):
        self.id = name
        self.raw_data = None
        self.repaired_data = None
        self.constraints = None
        self.aux_table = {}
        # Start DBengine.
        self.engine = DBengine(
            env['db_user'],
            env['db_pwd'],
            env['db_name'],
            env['db_host'],
            pool_size=env['threads'],
            timeout=env['timeout']
        )
        # Members to convert (tuple_id, attribute) to cell_id.
        self.attr_to_idx = {}
        self.attr_count = 0
        # Dataset statistics.
        self.stats_ready = None
        # Number of tuples.
        self.total_tuples = 0
        # Statistics for single attributes.
        self.single_attr_stats = {}
        # Statistics for attribute pairs.
        self.pair_attr_stats = {}
        # Conditional entropy of each pair of attributes using the log base 2.
        self.cond_entropies_base_2 = {}
        # Correlations between attributes
        self.correlations = None
        # Boolean flag for incremental behavior.
        self.incremental = env['incremental']
        # Boolean flag for compute entropy using the incremental algorithm.
        self.incremental_entropy = env['incremental_entropy']

    def load_data(self, name, fpath, na_values=None, entity_col=None, src_col=None):
        """
        load_data takes a CSV file of the initial data, adds tuple IDs (_tid_) to each row,
        and generates unique index numbers for each column.

        Creates a table with the user-supplied 'name' parameter (e.g. 'hospital').

        :param name: (str) name to initialize initial data with.
        :param fpath: (str) path to CSV file.
        :param na_values: (str) value that identifies a NULL value.
        :param entity_col: (str) column containing the unique identifier of an entity.
            For fusion tasks, rows with the same ID will be fused together in the output.
            If None, assumes every row is a unique entity.
        :param src_col: (str) if not None, for fusion tasks,
            specifies the column containing the source for each "mention" of an entity.
        """
        tic = time.clock()

        self.stats_ready = False

        try:
            # Do not include TID, and source column as trainable attributes.
            exclude_attr_cols = ['_tid_']
            if src_col is not None:
                exclude_attr_cols.append(src_col)

            # Load raw CSV file into a table 'name' (param) in PostgreSQL.
            self.raw_data = Table(name,
                                  Source.FILE,
                                  na_values=na_values,
                                  exclude_attr_cols=exclude_attr_cols,
                                  fpath=fpath)

            df = self.raw_data.df

            # Add '_tid_' column to dataset for uniquely identifying an entity.
            # If entity_col is not supplied, use auto-incrementing values.
            # Otherwise, we use the entity values directly as _tid_'s.
            if entity_col is None:
                df.insert(0, '_tid_', range(self.get_first_tid(), len(df.index)))
            else:
                df.rename({entity_col: '_tid_'}, axis='columns', inplace=True)

            # Use NULL_REPR to represent NULL values.
            df.fillna(NULL_REPR, inplace=True)

            logging.info("Loaded %d rows with %d cells",
                         len(df.index),
                         len(df.index) * len(df.columns))

            self.raw_data.store_to_db(self.engine.engine)
            status = 'DONE Loading {fname}'.format(fname=os.path.basename(fpath))

            # Generate indexes on attribute columns for faster queries.
            for attr in self.raw_data.get_attributes():
                self.raw_data.create_db_index(self.engine, [attr])

            # Create attr_to_idx dictionary, which assigns a unique index to each attribute,
            # and attr_count (total # of attributes).
            self.attr_to_idx = {attr: idx for idx, attr in enumerate(self.raw_data.get_attributes())}
            self.attr_count = len(self.attr_to_idx)
        except Exception:
            logging.error('Loading data for table %s', name)
            raise
        toc = time.clock()
        load_time = toc - tic
        return status, load_time

    def set_constraints(self, constraints):
        self.constraints = constraints

    def generate_aux_table(self, aux_table, df, store=False, index_attrs=False, append=False):
        """
        generate_aux_table writes/overwrites/appends to the auxiliary table specified by 'aux_table'.

        It does the following:
          1. stores/replaces the specified aux_table into Postgres (store=True), AND/OR
          2. sets an index on the aux_table's internal Pandas DataFrame (index_attrs=[<columns>]), AND/OR
          3. creates Postgres indexes for aux_table (store=True and index_attrs=[<columns>]), OR
          4. appends the specified aux_table to Postgres table

        :param aux_table: (AuxTable) auxiliary table to generate.
        :param df: (DataFrame) dataframe to memoize/store for this auxiliary table.
        :param store: (bool) if true, creates/replaces Postgres table for this auxiliary table.
        :param index_attrs: (list[str]) list of attributes to create indexes on.
        :param append: (bool) if true, appends this auxiliary table to Postgres table.
        """
        try:
            self.aux_table[aux_table] = Table(aux_table.name,
                                              Source.DF,
                                              df=df)

            if store:
                self.aux_table[aux_table].store_to_db(self.engine.engine)
            if index_attrs:
                self.aux_table[aux_table].create_df_index(index_attrs)
            if store and index_attrs:
                self.aux_table[aux_table].create_db_index(self.engine, index_attrs)
            if append:
                self.aux_table[aux_table].store_to_db(self.engine.engine, if_exists='append')
        except Exception:
            logging.error('Generating aux_table %s', aux_table.name)
            raise

    def generate_aux_table_sql(self, aux_table, query, index_attrs=None):
        """
        :param aux_table: (AuxTable) auxiliary table to generate.
        :param query: (str) SQL query whose result is used for generating the auxiliary table.
        :param index_attrs: (list[str]) list of attributes (columns) to create index on.
        """
        try:
            self.aux_table[aux_table] = Table(aux_table.name,
                                              Source.SQL,
                                              table_query=query,
                                              db_engine=self.engine)

            if index_attrs is not None:
                self.aux_table[aux_table].create_df_index(index_attrs)
                self.aux_table[aux_table].create_db_index(self.engine, index_attrs)
        except Exception:
            logging.error('Generating aux_table %s', aux_table.name)
            raise

    def get_raw_data(self):
        """
        get_raw_data returns a pandas.DataFrame containing the raw data as it was initially loaded.
        """
        if self.raw_data is None:
            raise Exception('ERROR No dataset loaded')
        return self.raw_data.df

    def get_attributes(self):
        """
        get_attributes returns the trainable/learnable attributes (i.e. excluding meta-columns such as '_tid_').
        """
        if self.raw_data is None:
            raise Exception('ERROR No dataset loaded')
        return self.raw_data.get_attributes()

    def get_cell_id(self, tuple_id, attr_name):
        """
        get_cell_id returns cell ID: a unique ID for every cell.

        Cell ID: _tid_ * (# of attributes) + attr_idx
        """
        vid = tuple_id * self.attr_count + self.attr_to_idx[attr_name]
        return vid

    def get_statistics(self):
        """
        Returns the statistics computed in the 'collect_stats' method.
        """
        if not self.stats_ready:
            logging.debug('Computing frequency, co-occurrence, and correlation statistics from raw data...')
            tic = time.clock()
            self.collect_stats()
            logging.debug('DONE computing statistics in %.2fs', time.clock() - tic)

        stats = (self.total_tuples, self.single_attr_stats, self.pair_attr_stats)
        self.stats_ready = True

        return stats

    def get_correlations(self):
        """
        Compute, if not ready, and return attribute correlations based on conditional entropy.
        :return:
        """
        if not self.stats_ready:
            logging.debug('Computing frequency, co-occurrence, and correlation statistics from raw data...')
            tic = time.clock()
            self.collect_stats()
            logging.debug('DONE computing statistics in %.2fs', time.clock() - tic)

        self.stats_ready = True

        return self.correlations

    def collect_stats(self):
        """
        Computes the following statistics:
          1. self.total_tuples (total # of tuples in the incoming data)
          2. self.single_attr_stats ({ attribute -> { value -> count } })
            the frequency (# of entities) of a given attribute-value
          3. self.pair_attr_stats ({ attr1 -> { attr2 -> { val1 -> { val2 -> count } } } })
            the statistics for each pair of attributes, attr1 and attr2, where:
              <attr1>: first attribute
              <attr2>: second attribute
              <val1>: value of <attr1>
              <val2>: value of <attr2> that appears at least once with <val1>
              <count>: frequency (# of entities) where attr1=val1 AND attr2=val2
            Also known as co-occurrence count.
        """
        tic = time.clock()

        total_tuples_loaded = None
        single_attr_stats_loaded = None
        pair_attr_stats_loaded = None

        if self.incremental:
            logging.debug('Loading existing statistics from the database...')
            total_tuples_loaded, single_attr_stats_loaded, pair_attr_stats_loaded = self.load_stats()

        logging.debug('Computing frequency and co-occurrence statistics from raw data...')

        # We get the statistics from the incoming data.
        data_df = self.get_raw_data()

        # Total number of tuples.
        self.total_tuples = len(data_df.index)

        # Single statistics.
        for attr in self.get_attributes():
            self.single_attr_stats[attr] = self.get_stats_single(attr)

        # Pairwise statistics.
        for cond_attr in self.get_attributes():
            self.pair_attr_stats[cond_attr] = {}

            for trg_attr in self.get_attributes():
                if cond_attr != trg_attr:
                    self.pair_attr_stats[cond_attr][trg_attr] = self.get_stats_pair(cond_attr,
                                                                                    trg_attr)

        Stats = namedtuple('Stats', 'total_tuples single_attr_stats pair_attr_stats')
        stats1 = Stats(total_tuples_loaded, single_attr_stats_loaded, pair_attr_stats_loaded)
        stats2 = Stats(self.total_tuples, self.single_attr_stats, self.pair_attr_stats)

        if (not self.incremental) or (single_attr_stats_loaded is None):
            # If any of the '*_loaded' variables is None, it means there were no previous statistics in the database.
            self.correlations = self.compute_norm_cond_entropy_corr()
        else:
            if self.incremental_entropy:
                # The incremental entropy calculation requires separate statistics.
                self.correlations = self.compute_norm_cond_entropy_corr_incremental(total_tuples_loaded,
                                                                                    single_attr_stats_loaded,
                                                                                    pair_attr_stats_loaded)

                # Merges statistics from incoming data to the loaded statistics.
                self.merge_stats(stats1, stats2)
            else:
                # Merges statistics from incoming data to the loaded statistics before computing entropy.
                self.merge_stats(stats1, stats2)

                self.correlations = self.compute_norm_cond_entropy_corr()

            # Memoizes merged statistics in class variables.
            self.total_tuples = stats1.total_tuples
            self.single_attr_stats = stats1.single_attr_stats
            self.pair_attr_stats = stats1.pair_attr_stats

        logging.debug('DONE computing statistics in %.2f secs', time.clock() - tic)
        self.save_stats()

    # noinspection PyMethodMayBeStatic
    def get_stats_single(self, attr):
        """
        Returns a dictionary where the keys are domain values for :param attr: and
        the values contain the frequency count of that value for this attribute.
        """
        # need to decode values into unicode strings since we do lookups via
        # unicode strings from Postgres
        data_df = self.get_raw_data()

        # TODO: Add a not null check everywhere needed as NULLs are now part of the statistics because of entropy
        # return data_df[[attr]].loc[data_df[attr] != NULL_REPR].groupby([attr]).size().to_dict()
        return data_df[[attr]].groupby([attr]).size().to_dict()

    # noinspection PyMethodMayBeStatic
    def get_stats_pair(self, first_attr, second_attr):
        """
        Returns a dictionary {first_val -> {second_val -> count } } where:
            <first_val>: all possible values for first_attr
            <second_val>: all values for second_attr that appear at least once with <first_val>
            <count>: frequency (# of entities) where first_attr=<first_val> AND second_attr=<second_val>
        Filters out NULL values so no entries in the dictionary would have NULLs.
        """
        data_df = self.get_raw_data()

        # TODO: Add a not null check everywhere needed as NULLs are now part of the statistics because of entropy
        # tmp_df = data_df[[first_attr, second_attr]]\
        #     .loc[(data_df[first_attr] != NULL_REPR) & (data_df[second_attr] != NULL_REPR)]\
        #     .groupby([first_attr, second_attr])\
        #     .size()\
        #     .reset_index(name="count")
        tmp_df = data_df[[first_attr, second_attr]]\
            .groupby([first_attr, second_attr])\
            .size()\
            .reset_index(name="count")
        return dictify_df(tmp_df)

    def merge_stats(self, stats1, stats2):
        """
        Concatenates statistics from stats2 to stats1

        :param stats1: (tuple) ...
        :param stats2: (tuple) ...
        """

        attrs = self.get_attributes()

        for attr in attrs:
            for val, count in stats2.single_attr_stats[attr].items():
                if val in stats1.single_attr_stats[attr].keys():
                    # The key 'val' already exists, so we just update the count.
                    stats1.single_attr_stats[attr][val] += count
                else:
                    # The key 'val' is new, so we insert a new dictionary for 'attr'.
                    stats1.single_attr_stats[attr].update({val: count})

        for cond_attr in attrs:
            for trg_attr in attrs:
                if cond_attr != trg_attr:
                    for cond_val in stats2.pair_attr_stats[cond_attr][trg_attr].keys():
                        for trg_val, count in stats2.pair_attr_stats[cond_attr][trg_attr][cond_val].items():
                            if cond_val in stats1.pair_attr_stats[cond_attr][trg_attr].keys():
                                if trg_val in stats1.pair_attr_stats[cond_attr][trg_attr][cond_val].keys():
                                    stats1.pair_attr_stats[cond_attr][trg_attr][cond_val][trg_val] += count
                                else:
                                    new_dict = {trg_val: count}
                                    stats1.pair_attr_stats[cond_attr][trg_attr][cond_val].update(new_dict)
                            else:
                                new_dict = {cond_val: {trg_val: count}}
                                stats1.pair_attr_stats[cond_attr][trg_attr].update(new_dict)

    # noinspection PyUnresolvedReferences
    def get_domain_info(self):
        """
        get_domain_info returns (number of random variables, count of distinct values across all attributes).
        """
        query = 'SELECT count(_vid_), max(domain_size) FROM %s' % AuxTables.cell_domain.name
        res = self.engine.execute_query(query)
        total_vars = int(res[0][0])
        classes = int(res[0][1])
        return total_vars, classes

    # noinspection PyUnresolvedReferences,PyTypeChecker
    def get_inferred_values(self):
        tic = time.clock()

        # Index of 'domain' equals inferred_val_idx + 1 because SQL arrays begin at index 1.
        query = "SELECT t1._tid_, t1.attribute, domain[inferred_val_idx + 1] as rv_value " \
                "FROM " \
                "(SELECT _tid_, attribute, _vid_, init_value, " \
                "string_to_array(regexp_replace(domain, \'[{\"\"}]\', \'\', \'gi\'), \'|||\') as domain " \
                "FROM %s) as t1, %s as t2 " \
                "WHERE t1._vid_ = t2._vid_" % (AuxTables.cell_domain.name, AuxTables.inf_values_idx.name)

        self.generate_aux_table_sql(AuxTables.inf_values_dom, query, index_attrs=['_tid_'])
        self.aux_table[AuxTables.inf_values_dom].create_db_index(self.engine, ['attribute'])

        status = "DONE collecting the inferred values."

        toc = time.clock()
        total_time = toc - tic

        return status, total_time

    def get_repaired_dataset(self):
        tic = time.clock()

        init_records = self.raw_data.df.sort_values(['_tid_']).to_records(index=False)
        t = self.aux_table[AuxTables.inf_values_dom]
        repaired_vals = dictify_df(t.df.reset_index())

        for tid in repaired_vals:
            for attr in repaired_vals[tid]:
                init_records[tid][attr] = repaired_vals[tid][attr]

        repaired_df = pd.DataFrame.from_records(init_records)
        name = self.raw_data.name + '_repaired'

        self.repaired_data = Table(name, Source.DF, df=repaired_df)
        self.repaired_data.store_to_db(self.engine.engine)

        status = "DONE generating repaired dataset."

        toc = time.clock()
        total_time = toc - tic

        return status, total_time

    def compute_norm_cond_entropy_corr(self):
        """
        Computes the correlations between attributes by calculating the normalized conditional entropy between them.
        The conditional entropy is asymmetric, therefore we need pairwise computations.

        The computed correlations are stored in a dictionary in the format:
        {
          attr_a: { cond_attr_i: corr_strength_a_i,
                    cond_attr_j: corr_strength_a_j, ...},
          attr_b: { cond_attr_i: corr_strength_b_i, ...}
        }.

        :return a dictionary of correlations.
        """
        attrs = self.get_attributes()

        corr = {}
        for x in attrs:
            corr[x] = {}

            if x not in self.cond_entropies_base_2.keys():
                self.cond_entropies_base_2[x] = {}

            x_domain_size = len(self.single_attr_stats[x].keys())

            for y in attrs:
                # Set correlation to 0.0 if entropy of x is 1 (only one possible value).
                if x_domain_size == 1:
                    corr[x][y] = 0.0
                    continue

                # Set correlation to 1 for same attributes.
                if x == y:
                    corr[x][y] = 1.0
                    continue

                # Compute the conditional entropy H(x|y).
                # If H(x|y) = 0, then y determines x, i.e. y -> x.
                self.cond_entropies_base_2[x][y] = self._conditional_entropy(x, y)

                # Use the domain size of x as the log base for normalizing the conditional entropy.
                # The conditional entropy is 0 for strongly correlated attributes and 1 for independent attributes.
                # We reverse this to reflect the correlation.
                corr[x][y] = 1.0 - (self.cond_entropies_base_2[x][y] / np.log2(x_domain_size))

        return corr

    def compute_norm_cond_entropy_corr_incremental(self, num_tuples_loaded, single_attr_stats_loaded,
                                                   pair_attr_stats_loaded):
        """
        Computes the correlations between attributes by calculating the normalized conditional entropy between them.
        The conditional entropy is asymmetric, therefore we need pairwise computations.

        The computed correlations are stored in a dictionary in the format:
        {
          attr_a: { cond_attr_i: corr_strength_a_i,
                    cond_attr_j: corr_strength_a_j, ...},
          attr_b: { cond_attr_i: corr_strength_b_i, ...}
        }.

        :return a dictionary of correlations.
        """

        attrs = self.get_attributes()

        corr = {}
        for x in attrs:
            corr[x] = {}

            if x not in self.cond_entropies_base_2.keys():
                self.cond_entropies_base_2[x] = {}

            if single_attr_stats_loaded is not None:
                # Compute the number of unique values for the attribute regarding both loaded and current stats
                unique_x_values = set(single_attr_stats_loaded[x].keys() + self.single_attr_stats[x].keys())
                x_domain_size = len(unique_x_values)
            else:
                x_domain_size = len(self.single_attr_stats[x].keys())

            for y in attrs:
                # Set correlation to 0.0 if entropy of x is 1 (only one possible value).
                if x_domain_size == 1:
                    corr[x][y] = 0.0
                    continue

                # Set correlation to 1 for same attributes.
                if x == y:
                    corr[x][y] = 1.0
                    continue

                # Compute the conditional entropy H(x|y).
                # If H(x|y) = 0, then y determines x, i.e. y -> x.
                self.cond_entropies_base_2[x][y] = self._conditional_entropy_incremental(x, y, num_tuples_loaded,
                                                                                         single_attr_stats_loaded,
                                                                                         pair_attr_stats_loaded)

                # Use the domain size of x as the log base for normalizing the conditional entropy.
                # The conditional entropy is 0 for strongly correlated attributes and 1 for independent attributes.
                # We reverse this to reflect the correlation.
                corr[x][y] = 1.0 - (self.cond_entropies_base_2[x][y] / np.log2(x_domain_size))

        return corr

    def _conditional_entropy(self, x_attr, y_attr):
        """
        Computes the conditional entropy considering the log base 2.

        :param x_attr: (string) name of attribute X.
        :param y_attr: (string) name of attribute Y.

        :return: the conditional entropy of attributes X and Y using the log base 2.
        """
        xy_entropy = 0.0

        y_freq = {}

        for value, count in self.single_attr_stats[y_attr].items():
            y_freq[value] = count

        for x_value in self.single_attr_stats[x_attr].keys():
            for y_value, xy_freq in self.pair_attr_stats[x_attr][y_attr][x_value].items():
                p_xy = xy_freq / float(self.total_tuples)

                xy_entropy = xy_entropy - (p_xy * np.log2(xy_freq / float(y_freq[y_value])))

        return xy_entropy

    def _conditional_entropy_incremental(self, x_attr, y_attr, num_tuples_loaded, single_attr_stats_loaded,
                                         pair_attr_stats_loaded):
        """
        Incrementally computes the conditional entropy considering the log base 2.

        :param x_attr: (string) name of attribute X.
        :param y_attr: (string) name of attribute Y.
        TODO: complete the comment

        :return: the conditional entropy of attributes X and Y using the log base 2.
        """

        total = num_tuples_loaded + self.total_tuples
        xy_entropy = (num_tuples_loaded / float(total)) * self.cond_entropies_base_2[x_attr][y_attr]

        # Builds the dictionary for existing frequencies of Y including 0's for new values.
        y_old_freq = {}
        for y_value in self.single_attr_stats[y_attr].keys():
            if y_value in single_attr_stats_loaded[y_attr].keys():
                y_old_freq[y_value] = single_attr_stats_loaded[y_attr][y_value]
            else:
                y_old_freq[y_value] = 0

        # Builds the dictionary for existing co-occurrences of X and Y including 0's for new values.
        xy_old_freq = {}
        for x_value in self.pair_attr_stats[x_attr][y_attr].keys():
            xy_old_freq[x_value] = {}
            if x_value in pair_attr_stats_loaded[x_attr][y_attr].keys():
                for y_value in self.pair_attr_stats[x_attr][y_attr][x_value].keys():
                    if y_value in pair_attr_stats_loaded[x_attr][y_attr][x_value].keys():
                        xy_old_freq[x_value][y_value] = pair_attr_stats_loaded[x_attr][y_attr][x_value][y_value]
                    else:
                        xy_old_freq[x_value][y_value] = 0
            else:
                for y_value in self.pair_attr_stats[x_attr][y_attr][x_value].keys():
                    xy_old_freq[x_value][y_value] = 0

        # Updates the entropy regarding the new terms.
        for x_value in self.pair_attr_stats[x_attr][y_attr].keys():
            for y_value, xy_freq in self.pair_attr_stats[x_attr][y_attr][x_value].items():
                new_term = xy_freq / float(total)
                log_term_num = xy_old_freq[x_value][y_value] + xy_freq
                log_term_den = y_old_freq[y_value] + self.single_attr_stats[y_attr][y_value]

                xy_entropy = xy_entropy - (new_term * np.log2(log_term_num / float(log_term_den)))

        # Updates the entropy regarding old terms which might need to be removed.
        for x_value in self.pair_attr_stats[x_attr][y_attr].keys():
            for y_value, xy_freq in self.pair_attr_stats[x_attr][y_attr][x_value].items():
                if xy_old_freq[x_value][y_value] != 0:
                    old_term = xy_old_freq[x_value][y_value] / float(total)
                    log_term_num = 1.0 + (xy_freq / float(xy_old_freq[x_value][y_value]))
                    log_term_den = 1.0 + (self.single_attr_stats[y_attr][y_value] /
                                          float(y_old_freq[y_value]))

                    xy_entropy = xy_entropy - (old_term * np.log2(log_term_num / log_term_den))

        return xy_entropy

    def get_first_tid(self):
        first_tid = 0

        if self.incremental:
            # TODO: Fetch from the database
            pass

        return first_tid

    def load_stats(self):
        # query = "SELECT _vid_, _cid_, _tid_, attribute, a.rv_val, a.val_id FROM %s" % AuxTables.cell_domain.name
        # query += ", unnest(string_to_array(regexp_replace(domain, \'[{\"\"}]\', \'\', \'gi\'), \'|||\'))"
        # query += "WITH ORDINALITY a(rv_val, val_id)"
        #
        # self.ds.generate_aux_table_sql(AuxTables.pos_values, query, index_attrs=['_tid_', 'attribute'])
        #
        # tmp_df = data_df[[first_attr, second_attr]]\
        #     .groupby([first_attr, second_attr])\
        #     .size()\
        #     .reset_index(name="count")
        #
        # dictify_df(tmp_df)
        #
        # num_tuples = 0
        #
        num_tuples = None
        single_attr_stats = None
        pair_attr_stats = None

        return num_tuples, single_attr_stats, pair_attr_stats

    def save_stats(self):

        single_stats = []
        for attr1 in self.single_attr_stats.keys():
            attr1_stats = [(attr1, attr2, freq) for attr2, freq in self.single_attr_stats[attr1].items()]
            single_stats += attr1_stats

        single_stats_df = pd.DataFrame(columns=['attr', 'val', 'freq'], data=single_stats)
        self.generate_aux_table(AuxTables.single_attr_stats,
                                single_stats_df.sort_values(by=['attr', 'val']), store=True)

        pair_stats = []
        for attr1 in self.pair_attr_stats.keys():
            for attr2 in self.pair_attr_stats[attr1].keys():
                for val1 in self.pair_attr_stats[attr1][attr2].keys():
                    attr1_attr2_val1_val2_stats = [(attr1, attr2, val1, val2, freq)
                                                   for val2, freq in self.pair_attr_stats[attr1][attr2][val1].items()]
            pair_stats += attr1_attr2_val1_val2_stats

        pair_stats_df = pd.DataFrame(columns=['attr1', 'attr2', 'val1', 'val2', 'freq'], data=pair_stats)
        self.generate_aux_table(AuxTables.pair_attr_stats,
                                pair_stats_df.sort_values(by=['attr1', 'attr2', 'val1', 'val2']), store=True)

        logging.info('Okay')

    def get_total_tuples(self):
        return self.total_tuples
