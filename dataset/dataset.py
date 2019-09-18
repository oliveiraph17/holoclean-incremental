from enum import Enum
import logging
import os
import time

import numpy as np
import pandas as pd

from .dbengine import DBengine
from .table import Table, Source
from utils import dictify_df, NULL_REPR
from recordtype import recordtype
from math import log2
from pyitlib import discrete_random_variable as drv
# from timeit import default_timer as timer
# from statistics import mean


class AuxTables(Enum):
    c_cells        = 1
    dk_cells       = 2
    cell_domain    = 3
    pos_values     = 4
    cell_distr     = 5
    inf_values_idx = 6
    inf_values_dom = 7
    single_attr_stats = 8
    pair_attr_stats = 9
    training_cells = 10
    repaired_table_copy = 11


class CellStatus(Enum):
    NOT_SET        = 0
    WEAK_LABEL     = 1
    SINGLE_VALUE   = 2

class Dataset:
    """
    This class keeps all dataframes and tables for a HC session.
    """
    def __init__(self, name, env):
        self.env = env
        self.id = name
        self.raw_data = None
        # DataFrame with the incoming data appended to previous repaired data.
        self.raw_data_with_previous_repaired_data_df = None
        self.repaired_data = None
        # DataFrame with the rows from the previous dataset that are involved in violations.
        self.previous_dirty_rows_df = None
        self.constraints = None
        self.aux_table = {}
        for tab in AuxTables:
            self.aux_table[tab] = None
        # start dbengine
        self.engine = DBengine(
            env['db_user'],
            env['db_pwd'],
            env['db_name'],
            env['db_host'],
            pool_size=env['threads'],
            timeout=env['timeout']
        )
        # members to convert (tuple_id, attribute) to cell_id
        self.attr_to_idx = {}
        self.attr_count = 0
        # dataset statistics
        self.stats_ready = None
        # Total tuples
        self.total_tuples = 0
        # Domain stats for single attributes
        self.single_attr_stats = {}
        # Domain stats for attribute pairs
        self.pair_attr_stats = {}
        # Active attributes (attributes with errors)
        self._active_attributes = None
        # Attributes to train on
        self.train_attrs = env["train_attrs"]

        # Embedding model for learned embedding vectors of domain values and
        # tuple context
        self._embedding_model = None

        # Numerical attribute list, all strings
        self.numerical_attrs = None
        self.categorical_attrs = None

        self.quantized_data = None
        self.do_quantization = False

        # Conditional entropy of each pair of attributes using the log base 2.
        self.cond_entropies_base_2 = {}
        # Correlations between attributes.
        self.correlations = None
        # First _tid_ to be loaded.
        self.first_tid = None
        # Boolean flag for incremental behavior.
        self.incremental = env['incremental']
        # Boolean flag for computing entropy using the incremental algorithm.
        self.incremental_entropy = env['incremental_entropy']
        # Boolean flag for computing entropy using the method implemented in pyitlib.
        self.default_entropy = env['default_entropy']
        # Boolean flag for repairing previous errors.
        self.repair_previous_errors = env['repair_previous_errors']
        # Boolean flag for recomputing statistics and retraining model from scratch in incremental scenarios.
        self.recompute_from_scratch = env['recompute_from_scratch']

    # TODO(richardwu): load more than just CSV files
    def load_data(self, name, fpath, na_values=None, entity_col=None, src_col=None,
                  exclude_attr_cols=None, numerical_attrs=None, store_to_db=True):
        """
        load_data takes a CSV file of the initial data, adds tuple IDs (_tid_)
        to each row to uniquely identify an 'entity', and generates unique
        index numbers for each attribute/column.

        Creates a table with the user supplied 'name' parameter (e.g. 'hospital').

        :param name: (str) name to initialize dataset with.
        :param fpath: (str) filepath to CSV file.
        :param na_values: (str) value that identifies a NULL value
        :param entity_col: (str) column containing the unique
            identifier/ID of an entity.  For fusion tasks, rows with
            the same ID will be fused together in the output.
            If None, assumes every row is a unique entity.
        :param src_col: (str) if not None, for fusion tasks
            specifies the column containing the source for each "mention" of an
            entity.
        :param exclude_attr_cols:
        :param numerical_attrs:
        """
        tic = time.clock()

        self.stats_ready = False

        try:
            # Do not include TID and source column as trainable attributes
            if exclude_attr_cols is None:
                exclude_attr_cols = ['_tid_']
            else:
                exclude_attr_cols.append('_tid_')
            if src_col is not None:
                exclude_attr_cols.append(src_col)

            # Load raw CSV file/data into a Postgres table 'name' (param).
            self.raw_data = Table(name, Source.FILE, na_values=na_values,
                                  exclude_attr_cols=exclude_attr_cols, fpath=fpath)

            df = self.raw_data.df
            # Add _tid_ column to dataset that uniquely identifies an entity.
            # If entity_col is not supplied, use auto-incrementing values.
            # Otherwise we use the entity values directly as _tid_'s.
            if entity_col is None:
                self.first_tid = self.get_first_tid()
                df.insert(0, '_tid_', range(self.first_tid, self.first_tid + len(df.index)))
            else:
                # use entity IDs as _tid_'s directly
                df.rename({entity_col: '_tid_'}, axis='columns', inplace=True)

            self.numerical_attrs = numerical_attrs or []
            all_attrs = self.raw_data.get_attributes()
            self.categorical_attrs = [attr for attr in all_attrs if attr not in self.numerical_attrs]

            if store_to_db:
                # Now df is all in str type, make a copy of df and then
                # 1. replace the null values in categorical data
                # 2. make the numerical attrs as float
                # 3. store the correct type into db (categorical->str, numerical->float)
                df_correct_type = df.copy()
                for attr in self.categorical_attrs:
                    df_correct_type.loc[df_correct_type[attr].isnull(), attr] = NULL_REPR
                for attr in self.numerical_attrs:
                    df_correct_type[attr] =  df_correct_type[attr].astype(float)

                df_correct_type.to_sql(self.raw_data.name, self.engine.engine, if_exists='replace', index=False,
                                       index_label=None)

            # for df, which is all str
            # Use NULL_REPR to represent NULL values
            df.replace('', NULL_REPR, inplace=True)
            df.fillna(NULL_REPR, inplace=True)

            logging.info("Loaded %d rows with %d cells", self.raw_data.df.shape[0],
                         self.raw_data.df.shape[0] * self.raw_data.df.shape[1])

            # Call to store to database
            status = 'DONE Loading {fname}'.format(fname=os.path.basename(fpath))

            if store_to_db:
                # Generate indexes on attribute columns for faster queries
                for attr in self.raw_data.get_attributes():
                    # Generate index on attribute
                    self.raw_data.create_db_index(self.engine,[attr])

            # Create attr_to_idx dictionary (assign unique index for each attribute)
            # and attr_count (total # of attributes)
            self.attr_to_idx = {attr: idx for idx, attr in enumerate(self.raw_data.get_attributes())}
            self.attr_count = len(self.attr_to_idx)
        except Exception:
            logging.error('loading data for table %s', name)
            raise
        toc = time.clock()
        load_time = toc - tic
        return status, load_time

    def set_constraints(self, constraints):
        self.constraints = constraints

    def generate_aux_table(self, aux_table, df, store=False, index_attrs=False, append=False):
        """
        generate_aux_table writes/overwrites the auxiliary table specified by
        'aux_table'.

        It does:
          1. stores/replaces (store=True) or appends (append=True) the specified aux_table into Postgres (store=True), AND/OR
          2. sets an index on the aux_table's internal Pandas DataFrame (index_attrs=[<columns>]), AND/OR
          3. creates Postgres indexes for aux_table (store=True and index_attrs=[<columns>])

        :param aux_table: (AuxTable) auxiliary table to generate
        :param df: (DataFrame) dataframe to memoize/store for this auxiliary table
        :param store: (bool) if true, creates/replaces Postgres table for this auxiliary table
        :param index_attrs: (list[str]) list of attributes to create indexes on. If store is true,
        also creates indexes on Postgres table.
        :param append: (bool) if True, appends this auxiliary table to the PostgreSQL table.
        """
        try:
            self.aux_table[aux_table] = Table(aux_table.name, Source.DF, df=df)
            if store:
                if append:
                    self.aux_table[aux_table].store_to_db(self.engine.engine, if_exists='append')
                else:
                    self.aux_table[aux_table].store_to_db(self.engine.engine)
            if index_attrs:
                self.aux_table[aux_table].create_df_index(index_attrs)
            if store and index_attrs:
                self.aux_table[aux_table].create_db_index(self.engine, index_attrs)
        except Exception:
            logging.error('generating aux_table %s', aux_table.name)
            raise

    def generate_aux_table_sql(self, aux_table, query, index_attrs=False):
        """
        :param aux_table: (AuxTable) auxiliary table to generate
        :param query: (str) SQL query whose result is used for generating the auxiliary table.
        """
        try:
            self.aux_table[aux_table] = Table(aux_table.name, Source.SQL, table_query=query, db_engine=self.engine)
            if index_attrs:
                self.aux_table[aux_table].create_df_index(index_attrs)
                self.aux_table[aux_table].create_db_index(self.engine, index_attrs)
        except Exception:
            logging.error('generating aux_table %s', aux_table.name)
            raise

    def get_raw_data(self):
        """
        get_raw_data returns a pandas.DataFrame containing the raw data as it was initially loaded.
        """
        if self.raw_data is None:
            raise Exception('ERROR No dataset loaded')
        return self.raw_data.df

    def get_quantized_data(self):
        """
        get_quantized_data returns a pandas.DataFrame containing the data after quantization
        :return: the data after quantization in pandas.DataFrame
        """
        if self.quantized_data is None:
            raise Exception('ERROR No dataset quantized')
        return self.quantized_data.df

    def get_attributes(self):
        """
        get_attributes return the trainable/learnable attributes (i.e. exclude meta
        columns like _tid_).
        """
        if self.raw_data is None:
            raise Exception('ERROR No dataset loaded')
        return self.raw_data.get_attributes()

    def get_active_attributes(self):
        """
        get_active_attributes returns the attributes to be modeled.

        If infer_mode = 'dk', these attributes correspond only to attributes that contain at least
        one potentially erroneous cell. Otherwise all attributes are returned.

        If applicable, in the provided :param:`train_attrs` variable.
        """
        if self.train_attrs is None:
            self.train_attrs = self.get_attributes()

        if self.env['infer_mode'] == 'dk':
            if self._active_attributes is None:
                raise Exception('ERROR no active attributes loaded. Run error detection first.')
            attrs = self._active_attributes
        elif self.env['infer_mode'] == 'all':
            attrs = self.get_attributes()
        else:
            raise Exception('infer mode must be one of {dk, all}')

        return sorted([attr for attr in attrs if attr in self.train_attrs])

    def get_cell_id(self, tuple_id, attr_name):
        """
        get_cell_id returns cell ID: a unique ID for every cell.

        Cell ID: _tid_ * (# of attributes) + attr_idx
        """
        vid = tuple_id*self.attr_count + self.attr_to_idx[attr_name]
        return vid

    def get_statistics(self):
        """
        get_statistics returns:
            1. self.total_tuples (total # of tuples)
            2. self.single_attr_stats ({ attribute -> { value -> count } })
              the frequency (# of entities) of a given attribute-value
            3. self.pair_attr_stats ({ attr1 -> { attr2 -> {val1 -> {val2 -> count } } } })
              the statistics for each pair of attributes, attr1 and attr2, where:
                <attr1>: first attribute
                <attr2>: second attribute
                <val1>: all values of <attr1>
                <val2>: values of <attr2> that appear at least once with <val1>.
                <count>: frequency (# of entities) where attr1=val1 AND attr2=val2
        """
        if not self.stats_ready:
            self.collect_stats()
            self.stats_ready = True

        stats = (self.total_tuples, self.single_attr_stats, self.pair_attr_stats)

        return stats

    def get_correlations(self):
        """
        Compute, if not ready, and return attribute correlations based on conditional entropy.
        :return:
        """
        if not self.stats_ready:
            self.collect_stats()
            self.stats_ready = True

        return self.correlations

    def collect_stats(self):
        """
        collect_stats memoizes:
          1. self.single_attr_stats ({ attribute -> { value -> count } })
            the frequency (# of entities) of a given attribute-value
          2. self.pair_attr_stats ({ attr1 -> { attr2 -> {val1 -> {val2 -> count } } } })
            where DataFrame contains 3 columns:
              <attr1>: all possible values for attr1 ('val1')
              <attr2>: all values for attr2 that appeared at least once with <val1> ('val2')
              <count>: frequency (# of entities) where attr1: val1 AND attr2: val2
            Also known as co-occurrence count.
        """
        # These combinations of parameters should never happen, since they are conflicting.
        if not self.incremental and self.recompute_from_scratch:
            raise Exception('Inconsistent parameters: incremental=%r, recompute_from_scratch=%r.' %
                            (self.incremental, self.recompute_from_scratch))
        elif self.incremental_entropy and self.recompute_from_scratch:
            raise Exception('Inconsistent parameters: incremental_entropy=%r, recompute_from_scratch=%r.' %
                            (self.incremental_entropy, self.recompute_from_scratch))

        total_tuples_loaded = None
        single_attr_stats_loaded = None
        pair_attr_stats_loaded = None

        logging.debug('Computing frequency, co-occurrence, and correlation statistics from raw data.')

        if self.incremental and not self.is_first_batch() and not self.recompute_from_scratch:
            tic = time.clock()
            total_tuples_loaded, single_attr_stats_loaded, pair_attr_stats_loaded = self.load_stats()
            logging.debug('DONE loading existing statistics from the database in %.2f secs.', time.clock() - tic)

        tic = time.clock()

        if not self.is_first_batch() and self.recompute_from_scratch:
            # We get the statistics both from previous and incoming data.
            self.load_previous_repaired_data()
            data_df = self.raw_data_with_previous_repaired_data_df
        else:
            # We get the statistics from the incoming data.
            data_df = self.raw_data.df

        # Total number of tuples.
        self.total_tuples = len(data_df.index)

        # Single statistics.
        for attr in self.get_attributes():
            self.single_attr_stats[attr] = self.get_stats_single(attr)
        # Compute co-occurrence frequencies.
        for cond_attr in self.get_attributes():
            self.pair_attr_stats[cond_attr] = {}
            for trg_attr in self.get_attributes():
                if trg_attr != cond_attr:
                    self.pair_attr_stats[cond_attr][trg_attr] = self.get_stats_pair(cond_attr, trg_attr)

        Stats = recordtype('Stats', 'total_tuples single_attr_stats pair_attr_stats')
        stats1 = Stats(total_tuples_loaded, single_attr_stats_loaded, pair_attr_stats_loaded)
        stats2 = Stats(self.total_tuples, self.single_attr_stats, self.pair_attr_stats)

        entropy_tic = time.clock()

        # TODO(kaster): adjust the correlation computation to use correctly the quantized_data as indicated below.
        # data_df = self.ds.get_quantized_data() if self.do_quantization \
        #    else self.ds.get_raw_data()
        if not self.incremental or self.is_first_batch():
            correlations = self.compute_norm_cond_entropy_corr()
        else:
            if self.incremental_entropy:
                # The incremental entropy calculation requires separate statistics.
                correlations = self.compute_norm_cond_entropy_corr_incremental(total_tuples_loaded,
                                                                               single_attr_stats_loaded,
                                                                               pair_attr_stats_loaded)

                # Merges statistics from incoming data to the loaded statistics.
                self.merge_stats(stats1, stats2)

                # Memoizes merged statistics in class variables.
                self.total_tuples = stats1.total_tuples
                self.single_attr_stats = stats1.single_attr_stats
                self.pair_attr_stats = stats1.pair_attr_stats
            else:
                if not self.recompute_from_scratch:
                    # Merges statistics from incoming data to the loaded statistics before computing entropy.
                    self.merge_stats(stats1, stats2)

                    # Memoizes merged statistics in class variables.
                    self.total_tuples = stats1.total_tuples
                    self.single_attr_stats = stats1.single_attr_stats
                    self.pair_attr_stats = stats1.pair_attr_stats

                correlations = self.compute_norm_cond_entropy_corr()

        if self.env['debug_mode']:
            corrs_df = pd.DataFrame.from_dict(correlations, orient='columns')
            corrs_df.index.name = 'cond_attr'
            corrs_df.columns.name = 'attr'
            pd.set_option('display.max_columns', len(corrs_df.columns))
            pd.set_option('display.max_rows', len(corrs_df.columns))
            logging.debug("correlations:\n%s", corrs_df)
            logging.debug("summary of correlations:\n%s", corrs_df.describe())

        self.correlations = self._make_hashable(correlations)

        logging.debug('DONE computing entropy in %.2f secs.', time.clock() - entropy_tic)
        logging.debug('DONE computing statistics from incoming data in %.2f secs.', time.clock() - tic)

    def _make_hashable(self, corr_dict):
        if isinstance(corr_dict, dict):
            return tuple(sorted((k, self._make_hashable(v)) for k, v in corr_dict.items()))

        return corr_dict

    def get_stats_single(self, attr):
        """
        Returns a dictionary where the keys are domain values for :param attr: and
        the values contain the frequency count of that value for this attribute.
        """
        # need to decode values into unicode strings since we do lookups via
        # unicode strings from Postgres
        if not self.is_first_batch() and self.recompute_from_scratch:
            data_df = self.raw_data_with_previous_repaired_data_df
        else:
            data_df = self.get_quantized_data() if self.do_quantization else self.get_raw_data()
        return data_df[[attr]].groupby([attr]).size().to_dict()

    def get_stats_pair(self, first_attr, second_attr):
        """
        Returns a dictionary {first_val -> {second_val -> count } } where:
            <first_val>: all possible values for first_attr
            <second_val>: all values for second_attr that appear at least once with <first_val>
            <count>: frequency (# of entities) where first_attr=<first_val> AND second_attr=<second_val>
        """
        if not self.is_first_batch() and self.recompute_from_scratch:
            data_df = self.raw_data_with_previous_repaired_data_df
        else:
            data_df = self.get_quantized_data() if self.do_quantization else self.get_raw_data()
        tmp_df = data_df[[first_attr, second_attr]]\
            .groupby([first_attr, second_attr])\
            .size()\
            .reset_index(name="count")
        return dictify_df(tmp_df)

    def merge_stats(self, stats1, stats2):
        """
        Adds statistics from stats2 to stats1.
        They are handled via recordtypes, which work like namedtuples but are mutable.
        The statistics are: total_tuples, single_attr_stats, and pair_attr_stats.

        :param stats1: (recordtype) loaded statistics.
        :param stats2: (recordtype) incoming statistics.
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

        stats1.total_tuples += stats2.total_tuples

    def get_domain_info(self):
        """
        Returns (number of random variables, count of distinct values across all attributes).
        """
        # query = 'SELECT count(_vid_), max(domain_size) FROM %s'%AuxTables.cell_domain.name
        # res = self.engine.execute_query(query)
        # total_vars = int(res[0][0])
        # classes = int(res[0][1])
        # return total_vars, classes
        query = 'SELECT count(_vid_) FROM %s'%AuxTables.cell_domain.name
        res = self.engine.execute_query(query)
        total_vars = int(res[0][0])
        query = 'SELECT attribute, max(domain_size) FROM %s GROUP BY attribute' % AuxTables.cell_domain.name
        res = self.engine.execute_query(query)
        classes = {}
        for row in res:
            classes[row[0]] = int(row[1])
        return total_vars, classes

    def get_inferred_values(self):
        tic = time.clock()
        # index into domain with inferred_val_idx + 1 since SQL arrays begin at index 1.
        query = "SELECT t1._tid_, t1.attribute, domain[inferred_val_idx + 1] as rv_value " \
                "FROM " \
                "(SELECT _tid_, attribute, " \
                "_vid_, init_value, string_to_array(regexp_replace(domain, \'[{\"\"}]\', \'\', \'gi\'), \'|||\') as domain " \
                "FROM %s) as t1, %s as t2 " \
                "WHERE t1._vid_ = t2._vid_"%(AuxTables.cell_domain.name, AuxTables.inf_values_idx.name)
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
        name = self.raw_data.name+'_repaired'
        self.repaired_data = Table(name, Source.DF, df=repaired_df)
        self.repaired_data.store_to_db(self.engine.engine)
        status = "DONE generating repaired dataset"
        toc = time.clock()
        total_time = toc - tic
        return status, total_time

    def get_repaired_dataset_incremental(self):
        tic = time.clock()

        tid_to_idx = {}

        if self.is_first_batch():
            # Gets only rows from the current batch for repairing.
            # Therefore, the '_tid_' of each row is enough to index 'init_records'.
            init_records = self.raw_data.df.sort_values(['_tid_']).to_records(index=False)
        else:
            # In this case, we have to create a mapping between '_tid_' and 'idx' to index 'init_records'.
            if self.repair_previous_errors:
                # Gets dirty rows from previous batches and rows from the current batch for repairing.
                df_by_tid = pd.concat([self.previous_dirty_rows_df, self.raw_data.df]).sort_values(['_tid_'])

                # Drops the old index [0..n, 0..m], generated by concatenating both DataFrames of sizes n and m,
                # and creates a new index [0..n+m+1].
                df_by_tid.reset_index(drop=True, inplace=True)
            else:
                # Gets only rows from the current batch for repairing.
                df_by_tid = self.raw_data.df.sort_values(['_tid_'])

            init_records = df_by_tid.to_records(index=False)

            # Converts the index into a regular attribute.
            df_by_tid.reset_index(level=0, inplace=True)

            # Creates a dictionary which maps each '_tid_' to its respective 'idx'.
            tid_to_idx = df_by_tid[['index', '_tid_']].set_index('_tid_').to_dict()

        t = self.aux_table[AuxTables.inf_values_dom]
        inferred_values = dictify_df(t.df.reset_index())

        # Dictionary to keep track of previous dirty values that had their values changed after inference.
        updated_previous_values = {}
        max_previous_tid = 0

        if not self.is_first_batch() and self.repair_previous_errors:
            max_previous_tid = self.previous_dirty_rows_df['_tid_'].max(axis=0)

        for tid in inferred_values:
            if self.is_first_batch():
                idx = tid
            else:
                idx = tid_to_idx['index'][tid]

                if self.repair_previous_errors and tid <= max_previous_tid:
                    updated_previous_values[tid] = []

            for attr in inferred_values[tid]:
                # Checks if the value was repaired.
                if init_records[idx][attr] != inferred_values[tid][attr]:
                    if not self.recompute_from_scratch:
                        # Updates the statistics regarding the repaired value.
                        # This must be done before replacing the old value with the repaired one.
                        self.update_value_stats(init_records[idx],
                                                attr,
                                                init_records[idx][attr],
                                                inferred_values[tid][attr])

                    # Updates the record.
                    init_records[idx][attr] = inferred_values[tid][attr]

                    # Keeps track of cell that was changed.
                    if not self.is_first_batch() and self.repair_previous_errors and tid <= max_previous_tid:
                        updated_previous_values[tid].append([attr, inferred_values[tid][attr]])

        repaired_df = pd.DataFrame.from_records(init_records)
        repaired_table_name = self.raw_data.name + '_repaired'

        # Keeps track of the time spent to generate a copy of the current repaired table, if needed.
        repaired_table_generation_time = 0
        if not self.is_first_batch() and self.repair_previous_errors:
            tic_repaired_table_copy = time.clock()
            # Makes a copy of the current repaired table in order to properly compute the evaluation metrics later on.
            repaired_table_copy_sql = "SELECT * FROM %s" % (self.raw_data.name + "_repaired")
            self.generate_aux_table_sql(AuxTables.repaired_table_copy, repaired_table_copy_sql)
            repaired_table_generation_time = time.clock() - tic_repaired_table_copy

            # Updates previous rows in the database.
            self.persist_repaired_previous_rows(updated_previous_values, repaired_table_name)

            beginning = len(self.previous_dirty_rows_df)
            end = beginning + len(self.raw_data.df)
            repaired_incoming_rows_df = repaired_df.iloc[beginning:end]

            self.repaired_data = Table(repaired_table_name, Source.DF, df=repaired_incoming_rows_df)
            self.repaired_data.store_to_db(self.engine.engine, if_exists='append')
        else:
            self.repaired_data = Table(repaired_table_name, Source.DF, df=repaired_df)
            self.repaired_data.store_to_db(self.engine.engine)

        if not self.recompute_from_scratch:
            # Persists the up-to-date statistics in the database.
            tic_inc = time.clock()
            self.save_stats()
            logging.debug('DONE storing computed statistics in the database in %.2f secs.', time.clock() - tic_inc)

        if self.is_first_batch():
            # This index is useful for querying the maximum _tid_ value in get_first_tid().
            tic_inc = time.clock()
            self.repaired_data.create_db_index(self.engine, ['_tid_'])
            logging.debug('DONE indexing `_tid_` column on repaired table in %.2f secs.', time.clock() - tic_inc)

        status = "DONE generating repaired dataset."

        toc = time.clock()
        total_time = toc - tic - repaired_table_generation_time

        return status, total_time

    def persist_repaired_previous_rows(self, updated_previous_values, repaired_table_name):
        sql_commands = []

        for tid in updated_previous_values:
            # Checks if the list of updates for the current _tid_ is not empty.
            if updated_previous_values[tid]:
                command = "UPDATE {} SET".format(repaired_table_name)

                assignments = []
                for pair in updated_previous_values[tid]:
                    assignments.append(" \"{}\" = '{}'".format(pair[0], pair[1]))
                command += ",".join(assignments)

                command += " WHERE _tid_ = {}".format(tid)

                sql_commands.append(command)

        # Checks if the list of SQL commands is not empty.
        if sql_commands:
            self.engine.execute_updates(sql_commands)

    def update_value_stats(self, row, attr, old_attr_val, new_attr_val):
        # This if-else block removes/decrements the frequency of 'old_attr_val'.
        if self.single_attr_stats[attr][old_attr_val] == 1:
            # Removes it from the single-attribute statistics.
            self.single_attr_stats[attr].pop(old_attr_val)

            # All entries corresponding to 'old_attr_val'
            # must be removed from the pairwise statistics as well.
            for other_attr in self.get_attributes():
                if attr != other_attr:
                    # Co-occurring value of 'other_attr' in the current repaired tuple.
                    other_attr_val = row[other_attr]

                    # Between 'attr' and 'other_attr'.
                    self.pair_attr_stats[attr][other_attr].pop(old_attr_val)

                    # The other way around.
                    self.pair_attr_stats[other_attr][attr][other_attr_val].pop(old_attr_val)
        else:
            # Decrements it in the single-attribute statistics.
            self.single_attr_stats[attr][old_attr_val] -= 1

            # The frequency of 'old_attr_val' with each co-occurring value
            # in the current repaired tuple must be updated in the pairwise statistics.
            for other_attr in self.get_attributes():
                if attr != other_attr:
                    # Co-occurring value of 'other_attr' in the current repaired tuple.
                    other_attr_val = row[other_attr]

                    # Between 'attr' and 'other_attr'.
                    if self.pair_attr_stats[attr][other_attr][old_attr_val][other_attr_val] == 1:
                        self.pair_attr_stats[attr][other_attr][old_attr_val].pop(other_attr_val)
                    else:
                        self.pair_attr_stats[attr][other_attr][old_attr_val][other_attr_val] -= 1

                    # The other way around.
                    if self.pair_attr_stats[other_attr][attr][other_attr_val][old_attr_val] == 1:
                        self.pair_attr_stats[other_attr][attr][other_attr_val].pop(old_attr_val)
                    else:
                        self.pair_attr_stats[other_attr][attr][other_attr_val][old_attr_val] -= 1

        # Now we either create a new entry for the repaired value frequency or increment it.
        if self.single_attr_stats[attr].get(new_attr_val) is None:
            # The repaired value does not exist yet, so we create an entry for its frequency.
            self.single_attr_stats[attr][new_attr_val] = 1

            # The frequency of 'new_attr_val' with each co-occurring value
            # in the current repaired tuple must be created in the pairwise statistics.
            for other_attr in self.get_attributes():
                if attr != other_attr:
                    other_attr_val = row[other_attr]

                    # Adds the key 'new_attr_val' with the corresponding nested dictionary.
                    self.pair_attr_stats[attr][other_attr][new_attr_val] = {other_attr_val: 1}

                    # Adds the key 'new_attr_val' with the value 1.
                    self.pair_attr_stats[other_attr][attr][other_attr_val][new_attr_val] = 1
        else:
            # The repaired value already exists, so we increment its frequency.
            self.single_attr_stats[attr][new_attr_val] += 1

            # The frequency of 'new_attr_val' with each co-occurring value
            # in the current repaired tuple must be updated in the pairwise statistics.
            for other_attr in self.get_attributes():
                if attr != other_attr:
                    # Co-occurring value of 'other_attr' in the current repaired tuple.
                    other_attr_val = row[other_attr]

                    # Between 'attr' and 'other_attr'.
                    if self.pair_attr_stats[attr][other_attr].get(new_attr_val) is None:
                        # Adds the key 'new_attr_val' with the corresponding nested dictionary.
                        self.pair_attr_stats[attr][other_attr][new_attr_val] = {other_attr_val: 1}
                    elif self.pair_attr_stats[attr][other_attr][new_attr_val].get(other_attr_val) is None:
                        # Adds the key 'other_attr_val' with the value 1.
                        self.pair_attr_stats[attr][other_attr][new_attr_val][other_attr_val] = 1
                    else:
                        # Increments the frequency.
                        self.pair_attr_stats[attr][other_attr][new_attr_val][other_attr_val] += 1

                    # The other way around.
                    if self.pair_attr_stats[other_attr][attr][other_attr_val].get(new_attr_val) is None:
                        # Adds the key 'new_attr_val' with the value 1.
                        self.pair_attr_stats[other_attr][attr][other_attr_val][new_attr_val] = 1
                    else:
                        # Increments the frequency.
                        self.pair_attr_stats[other_attr][attr][other_attr_val][new_attr_val] += 1

    def compute_norm_cond_entropy_corr(self):
        """
        Computes the correlations between attributes by calculating the normalized conditional entropy between them.
        The conditional entropy is asymmetric, therefore we need pairwise computations.

        The computed correlations are stored in a dictionary in the format:
        {
          attr_a: { cond_attr_i: corr_strength_a_i,
                    cond_attr_j: corr_strength_a_j, ...},
          attr_b: { cond_attr_i: corr_strength_b_i, ...}
        }

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
                # Sets correlation to 1.0 for same attributes.
                if x == y:
                    corr[x][y] = 1.0
                    continue

                # Sets correlation to 0.0 if entropy of x is 1 (only one possible value).
                if x_domain_size == 1:
                    corr[x][y] = 0.0
                    continue

                # Computes the conditional entropy H(x|y).
                # If H(x|y) = 0, then y determines x, i.e. y -> x.
                if self.default_entropy:
                    # Uses the implementation of pyitlib.
                    corr[x][y] = 1.0 - drv.entropy_conditional(self.raw_data.df[x],
                                                               self.raw_data.df[y],
                                                               base=x_domain_size).item()
                else:
                    # Uses our implementation.
                    self.cond_entropies_base_2[x][y] = self._conditional_entropy(x, y)

                    # Uses the domain size of x as the log base for normalizing the conditional entropy.
                    # The conditional entropy is 0 for strongly correlated attributes and 1 for independent attributes.
                    # We reverse this to reflect the correlation.
                    corr[x][y] = 1.0 - (self.cond_entropies_base_2[x][y] / log2(x_domain_size))

        return corr

    def compute_norm_cond_entropy_corr_incremental(self,
                                                   num_tuples_loaded,
                                                   single_attr_stats_loaded,
                                                   pair_attr_stats_loaded):
        """
        Computes the correlations between attributes by calculating the normalized conditional entropy between them.
        The conditional entropy is asymmetric, therefore we need pairwise computations.
        Performs the computation incrementally.

        The computed correlations are stored in a dictionary in the format:
        {
          attr_a: { cond_attr_i: corr_strength_a_i,
                    cond_attr_j: corr_strength_a_j, ...},
          attr_b: { cond_attr_i: corr_strength_b_i, ...}
        }

        :return a dictionary of correlations.
        """

        attrs = self.get_attributes()

        corr = {}
        for x in attrs:
            corr[x] = {}

            if x not in self.cond_entropies_base_2.keys():
                self.cond_entropies_base_2[x] = {}

            # Computes the number of unique values for this attribute regarding both loaded and current statistics.
            unique_x_values = set(single_attr_stats_loaded[x].keys())
            unique_x_values.update(self.single_attr_stats[x].keys())

            x_domain_size = len(unique_x_values)

            for y in attrs:
                # Sets correlation to 1.0 for same attributes.
                if x == y:
                    corr[x][y] = 1.0
                    continue

                # Sets correlation to 0.0 if entropy of x is 1 (only one possible value).
                if x_domain_size == 1:
                    corr[x][y] = 0.0
                    continue

                # Computes the conditional entropy H(x|y).
                # If H(x|y) = 0, then y determines x, i.e. y -> x.
                self.cond_entropies_base_2[x][y] = self._conditional_entropy_incremental(x,
                                                                                         y,
                                                                                         num_tuples_loaded,
                                                                                         single_attr_stats_loaded,
                                                                                         pair_attr_stats_loaded)

                # Uses the domain size of x as the log base for normalizing the conditional entropy.
                # The conditional entropy is 0.0 for strongly correlated attributes and 1.0 for independent attributes.
                # We reverse this to reflect the correlation.
                corr[x][y] = 1.0 - (self.cond_entropies_base_2[x][y] / log2(x_domain_size))

        return corr

    def _conditional_entropy(self, x_attr, y_attr):
        """
        Computes the conditional entropy considering the log base 2.

        :param x_attr: (str) name of attribute X.
        :param y_attr: (str) name of attribute Y.

        :return: the conditional entropy of attributes X and Y using the log base 2.
        """
        xy_entropy = 0.0

        for x_value in self.single_attr_stats[x_attr].keys():
            for y_value, xy_freq in self.pair_attr_stats[x_attr][y_attr][x_value].items():
                p_xy = xy_freq / float(self.total_tuples)

                xy_entropy = xy_entropy - (p_xy * log2(xy_freq / float(self.single_attr_stats[y_attr][y_value])))

        return xy_entropy

    def _conditional_entropy_incremental(self,
                                         x_attr,
                                         y_attr,
                                         num_tuples_loaded,
                                         single_attr_stats_loaded,
                                         pair_attr_stats_loaded):
        """
        Incrementally computes the conditional entropy considering the log base 2.

        :param x_attr: (str) name of attribute X.
        :param y_attr: (str) name of attribute Y.
        :param num_tuples_loaded: (int) number of existing tuples in the database.
        :param single_attr_stats_loaded: (dict) single-attribute statistics of existing tuples.
        :param pair_attr_stats_loaded: (dict) pairwise statistics of existing tuples.

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

                xy_entropy = xy_entropy - (new_term * log2(log_term_num / float(log_term_den)))

        # Updates the entropy regarding old terms which might need to be removed.
        for x_value in self.pair_attr_stats[x_attr][y_attr].keys():
            for y_value, xy_freq in self.pair_attr_stats[x_attr][y_attr][x_value].items():
                if xy_old_freq[x_value][y_value] != 0:
                    old_term = xy_old_freq[x_value][y_value] / float(total)
                    log_term_num = 1.0 + (xy_freq / float(xy_old_freq[x_value][y_value]))
                    log_term_den = 1.0 + (self.single_attr_stats[y_attr][y_value] /
                                          float(y_old_freq[y_value]))

                    xy_entropy = xy_entropy - (old_term * log2(log_term_num / log_term_den))

        return xy_entropy

    def get_first_tid(self):
        first_tid = 0

        if self.incremental:
            try:
                table_repaired_name = self.raw_data.name + '_repaired'
                query = 'SELECT MAX(t1._tid_) FROM {} as t1'.format(table_repaired_name)
                result = self.engine.execute_query(query)
                first_tid = result[0][0] + 1
            except Exception:
                # There is no previously repaired table in the database.
                pass

        return first_tid

    def load_previous_repaired_data(self):
        try:
            previous_repaired_data = Table(self.raw_data.name + '_repaired',
                                           Source.DB,
                                           db_engine=self.engine)

            incoming_data_df = self.get_quantized_data() if self.do_quantization else self.get_raw_data()

            self.raw_data_with_previous_repaired_data_df = pd.concat([previous_repaired_data.df,
                                                                      incoming_data_df])
        except ValueError:
            raise Exception('ERROR while trying to load previous repaired data from the database.')

    def load_stats(self):
        num_tuples = None
        single_attr_stats = None
        pair_attr_stats = None

        try:
            self.aux_table[AuxTables.single_attr_stats] = Table(AuxTables.single_attr_stats.name,
                                                                Source.DB,
                                                                db_engine=self.engine)

            single_attr_stats = dictify_df(self.aux_table[AuxTables.single_attr_stats].df)

            self.aux_table[AuxTables.pair_attr_stats] = Table(AuxTables.pair_attr_stats.name,
                                                              Source.DB,
                                                              db_engine=self.engine)

            pair_attr_stats = dictify_df(self.aux_table[AuxTables.pair_attr_stats].df)

            table_repaired_name = self.raw_data.name + '_repaired'
            query = 'SELECT COUNT(*) FROM {}'.format(table_repaired_name)
            result = self.engine.execute_query(query)
            num_tuples = result[0][0]
        except ValueError:
            raise Exception('ERROR while trying to load statistics from the database.')

        return num_tuples, single_attr_stats, pair_attr_stats

    def save_stats(self):
        # For using an attribute table (attr_idx, attr_name)
        # attrs = pd.DataFrame(data=self.attr_to_idx.items(), columns=['attr_name', 'attr_idx'])
        # self.generate_aux_table(AuxTables.attrs, attrs, store=True)

        # TODO: Update only the values that changed instead of replacing the statistics entirely.
        single_stats = []
        for attr in self.single_attr_stats.keys():
            attr_stats = [(attr, val, freq) for val, freq in self.single_attr_stats[attr].items()]
            single_stats += attr_stats

        single_stats_df = pd.DataFrame(columns=['attr', 'val', 'freq'], data=single_stats)
        self.generate_aux_table(AuxTables.single_attr_stats,
                                single_stats_df.sort_values(by=['attr', 'val']),
                                store=True)

        pair_stats = []
        for attr1 in self.pair_attr_stats.keys():
            for attr2 in self.pair_attr_stats[attr1].keys():
                for val1 in self.pair_attr_stats[attr1][attr2].keys():
                    attr1_attr2_val1_val2_stats = [(attr1, attr2, val1, val2, freq)
                                                   for val2, freq in self.pair_attr_stats[attr1][attr2][val1].items()]
                    pair_stats += attr1_attr2_val1_val2_stats

        pair_stats_df = pd.DataFrame(columns=['attr1', 'attr2', 'val1', 'val2', 'freq'], data=pair_stats)
        self.generate_aux_table(AuxTables.pair_attr_stats,
                                pair_stats_df.sort_values(by=['attr1', 'attr2', 'val1', 'val2']),
                                store=True)

    def is_first_batch(self):
        # self.first_tid is set to 0 in 'load_data()' method if this is the first batch of data.
        return self.first_tid == 0

    def set_previous_dirty_rows(self, previous_dirty_rows_df):
        self.previous_dirty_rows_df = previous_dirty_rows_df

    def get_previous_dirty_rows(self):
        if self.previous_dirty_rows_df is None:
            raise Exception('ERROR: Potentially dirty rows from the previous dataset could not be loaded.')

        return self.previous_dirty_rows_df

    def load_embedding_model(self, model):
        """
        Memoize the TupleEmbedding model for retrieving learned embeddings
        later (e.g. in EmbeddingFeaturizer).
        """
        self._embedding_model = model

    def get_embedding_model(self):
        """
        Retrieve the memoized embedding model.
        """
        if self._embedding_model is None:
            raise Exception("cannot retrieve embedding model: it was never trained and loaded!")
        return self._embedding_model

