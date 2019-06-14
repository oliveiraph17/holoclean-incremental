from enum import Enum
import logging
import os
import time

import pandas as pd

from .dbengine import DBengine
from .table import Table, Source
from utils import dictify_df, NULL_REPR


class AuxTables(Enum):
    c_cells = 1
    dk_cells = 2
    cell_domain = 3
    pos_values = 4
    cell_distr = 5
    inf_values_idx = 6
    inf_values_dom = 7


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
        self.new_data = None
        self.repaired_data = None
        self.constraints = None
        self.aux_table = {}
        for tab in AuxTables:
            self.aux_table[tab] = None
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
        # Number of tuples.
        self.raw_total = 0
        self.new_total = 0;
        # Domain statistics for single attributes (excluding NULLs).
        self.single_attr_stats = {}
        # Conditional entropy statistics for single attributes (including NULLs).
        self.single_attr_stats_w_nulls = {}
        # Domain statistics for attribute pairs (excluding NULLs).
        self.pair_attr_stats = {}
        # Conditional entropy statistics for attribute pairs (including NULLs).
        self.pair_attr_stats_w_nulls = {}
        # Conditional entropy incremental statistics for single attributes (including NULLs).
        self.inc_single_attr_stats_w_nulls = {}
        # Conditional entropy incremental statistics for attribute pairs (including NULLs).
        self.inc_pair_attr_stats_w_nulls = {}

    # TODO(richardwu): load more than just CSV files
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
        try:
            # Do not include TID, batch number, and source column as trainable attributes.
            exclude_attr_cols = ['_tid_', '_batch_']
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
                df.insert(0, '_tid_', range(0, len(df)))
            else:
                df.rename({entity_col: '_tid_'}, axis='columns', inplace=True)

            # Use NULL_REPR to represent NULL values.
            df.fillna(NULL_REPR, inplace=True)

            # First batch of data, hence the values being 1.
            df['_batch_'] = [1] * len(df.index)

            logging.info("Loaded %d rows with %d cells",
                         len(df.index),
                         len(df.index) * self.raw_data.df.shape[1])

            self.raw_data.store_to_db(self.engine.engine)
            status = 'DONE Loading {fname}'.format(fname=os.path.basename(fpath))

            # Generate indexes on attribute columns for faster queries.
            for attr in self.raw_data.get_attributes():
                self.raw_data.create_db_index(self.engine, [attr])

            # Create index on _batch_ even though it is not trainable.
            # This is useful for the violation detector's queries.
            self.raw_data.create_db_index(self.engine, ['_batch_'])

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

    def load_new_data(self, name, fpath, batch, na_values=None, entity_col=None, src_col=None):
        """
        load_new_data takes a CSV file of the incoming data and adds tuple IDs (_tid_) to each row.
        It does not generates unique index numbers for each column, since this is previously done in load_data.

        Creates a table with the user-supplied 'name' parameter (e.g. 'hospital').

        :param name: (str) name to initialize incoming data with.
        :param fpath: (str) path to CSV file.
        :param batch: (int) greater than 1, regarding the batch number of the current incoming data
        :param na_values: (str) value that identifies a NULL value.
        :param entity_col: (str) column containing the unique identifier of an entity.
            For fusion tasks, rows with the same ID will be fused together in the output.
            If None, assumes every row is a unique entity.
        :param src_col: (str) if not None, for fusion tasks,
            specifies the column containing the source for each "mention" of an entity.
        """

        tic = time.clock()
        try:
            # There is already a raw_data and a new_data.
            # Therefore, we concatenate them so they become raw_data.
            # Then, the current incoming data will become new_data.
            if batch > 2:
                self.raw_data.df = pd.concat([self.raw_data.df, self.new_data.df],
                                             axis=0,
                                             ignore_index=True).reset_index(drop=True)

            # Do not include TID, batch number, and source column as trainable attributes.
            exclude_attr_cols = ['_tid_', '_batch_']
            if src_col is not None:
                exclude_attr_cols.append(src_col)

            self.new_data = Table(name,
                                  Source.FILE,
                                  na_values=na_values,
                                  exclude_attr_cols=exclude_attr_cols,
                                  fpath=fpath)

            df = self.new_data.df

            if entity_col is None:
                # Generates _tid_'s starting from the number of elements previously loaded.
                df.insert(0, '_tid_', range(len(self.raw_data.df.index),
                                            len(self.raw_data.df.index) + len(df.index)))
            else:
                df.rename({entity_col: '_tid_'}, axis='columns', inplace=True)

            df.fillna(NULL_REPR, inplace=True)

            df['_batch_'] = [batch] * len(df.index)

            logging.info("Loaded %d rows with %d cells",
                         len(df.index),
                         len(df.index) * self.new_data.df.shape[1])

            # The new data are appended to the existing data.
            # Therefore, we do not index the attributes afterwards.
            # They have already been indexed previously, and the indexes are updated automatically by the DBMS.
            self.new_data.store_to_db(self.engine.engine, if_exists='append')
            status = 'DONE Loading {fname}'.format(fname=os.path.basename(fpath))
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

    def generate_aux_table_sql(self, aux_table, query, index_attrs=False):
        """
        :param aux_table: (AuxTable) auxiliary table to generate.
        :param query: (str) SQL query whose result is used for generating the auxiliary table.
        :param index_attrs: (bool) if true, indexes attributes in the DataFrame and in the table.
        """
        try:
            self.aux_table[aux_table] = Table(aux_table.name,
                                              Source.SQL,
                                              table_query=query,
                                              db_engine=self.engine)

            if index_attrs:
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

    def get_new_data(self):
        """
        get_new_data returns a pandas.DataFrame containing the incoming data as it was initially loaded.
        """
        if self.new_data is None:
            raise Exception('ERROR No incoming data loaded')
        return self.new_data.df

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

    def get_statistics(self, batch=1):
        """
        get_statistics returns:
          1. self.raw_total (total # of tuples in the first batch of data)
          2. self.new_total (total # of tuples in the incoming data)
          3. self.single_attr_stats ({ attribute -> { value -> count } })
            the frequency (# of entities) of a given attribute-value
          4. self.single_attr_stats_w_nulls ({ attribute -> { value -> count } })
            same as 'single_attr_stats', but including counts for NULLs
          5. self.pair_attr_stats ({ attr1 -> { attr2 -> { val1 -> { val2 -> count } } } })
            the statistics for each pair of attributes, attr1 and attr2, where:
              <attr1>: first attribute
              <attr2>: second attribute
              <val1>: value of <attr1>
              <val2>: value of <attr2> that appears at least once with <val1>
              <count>: frequency (# of entities) where attr1=val1 AND attr2=val2
          6. self.pair_attr_stats_w_nulls ({ attr1 -> { attr2 -> { val1 -> { val2 -> count } } } })
            same as 'pair_attr_stats', but including counts for NULLs

        Neither 'single_attr_stats' nor 'pair_attr_stats' contain frequencies NULL values (NULL_REPR).
        One would need to explicitly check if the value is NULL before lookup.
        Also, values that only co-occur with NULLs will NOT be in 'pair_attr_stats'.
        """
        if batch == 1:
            logging.debug('Computing frequency and co-occurrence statistics from raw data...')
        else:
            logging.debug('Computing frequency and co-occurrence statistics from new data, batch %d...', batch)

        tic = time.clock()
        self.collect_stats(batch)
        logging.debug('DONE computing statistics in %.2fs', time.clock() - tic)

        stats = (self.raw_total,
                 self.new_total,
                 self.single_attr_stats,
                 self.single_attr_stats_w_nulls,
                 self.pair_attr_stats,
                 self.pair_attr_stats_w_nulls,
                 self.inc_single_attr_stats_w_nulls,
                 self.inc_pair_attr_stats_w_nulls)

        return stats

    def collect_stats(self, batch=1):
        logging.debug("Collecting single/pairwise statistics...")

        if batch == 1:
            # First batch of data.
            # We get the statistics from the initial data, assigning them for the first time.
            data_df = self.get_raw_data()

            # Total number of tuples.
            self.raw_total = data_df.shape[0]

            # Single statistics.
            for attr in self.get_attributes():
                self.single_attr_stats[attr] = self.get_stats_single(attr, data_df)
                self.single_attr_stats_w_nulls[attr] = self.get_stats_single_w_nulls(attr, data_df)

            # Pairwise statistics.
            for cond_attr in self.get_attributes():
                self.pair_attr_stats[cond_attr] = {}
                self.pair_attr_stats_w_nulls[cond_attr] = {}

                for trg_attr in self.get_attributes():
                    if cond_attr != trg_attr:
                        self.pair_attr_stats[cond_attr][trg_attr] = self.get_stats_pair(cond_attr,
                                                                                        trg_attr,
                                                                                        data_df)
                        self.pair_attr_stats_w_nulls[cond_attr][trg_attr] = self.get_stats_pair_w_nulls(cond_attr,
                                                                                                        trg_attr,
                                                                                                        data_df)
        else:
            # New batch of data.
            # We get the statistics from the incoming data, adding them to the existing statistics.
            data_df = self.get_new_data()

            # Total number of incoming tuples.
            self.new_total = data_df.shape[0]

            # Update single statistics.
            for attr in self.get_attributes():
                for val, count in self.get_stats_single(attr, data_df).items():
                    if val in self.single_attr_stats[attr]:
                        # The key 'val' already exists, so we just update the count.
                        self.single_attr_stats[attr][val] += count
                    else:
                        # The key 'val' is new, so we insert a new dictionary for 'attr'.
                        self.single_attr_stats[attr].update({val: count})

                self.inc_single_attr_stats_w_nulls[attr] = self.get_stats_single_w_nulls(attr, data_df)

            # Update pairwise statistics.
            for cond_attr in self.get_attributes():
                self.inc_pair_attr_stats_w_nulls[cond_attr] = {}

                for trg_attr in self.get_attributes():
                    if cond_attr != trg_attr:
                        # Statistics excluding NULLs, which will be used in the domain generation.
                        for cond_val, nested_dict in self.get_stats_pair(cond_attr, trg_attr, data_df).items():
                            for trg_val, count in nested_dict.items():
                                if cond_val in self.pair_attr_stats[cond_attr][trg_attr]:
                                    if trg_val in self.pair_attr_stats[cond_attr][trg_attr][cond_val]:
                                        self.pair_attr_stats[cond_attr][trg_attr][cond_val][trg_val] += count
                                    else:
                                        self.pair_attr_stats[cond_attr][trg_attr][cond_val].update({trg_val: count})
                                else:
                                    self.pair_attr_stats[cond_attr][trg_attr].update({cond_val: {trg_val: count}})

                        # Statistics including NULLs, which will be used in the conditional entropy computation.
                        self.inc_pair_attr_stats_w_nulls[cond_attr][trg_attr] = self.get_stats_pair_w_nulls(cond_attr,
                                                                                                            trg_attr,
                                                                                                            data_df)

    # noinspection PyMethodMayBeStatic
    def get_stats_single(self, attr, data_df):
        return data_df[[attr]].loc[data_df[attr] != NULL_REPR].groupby([attr]).size().to_dict()

    # noinspection PyMethodMayBeStatic
    def get_stats_single_w_nulls(self, attr, data_df):
        return data_df[[attr]].groupby([attr]).size().to_dict()

    # noinspection PyMethodMayBeStatic
    def get_stats_pair(self, first_attr, second_attr, data_df):
        tmp_df = data_df[[first_attr, second_attr]]\
            .loc[(data_df[first_attr] != NULL_REPR) & (data_df[second_attr] != NULL_REPR)]\
            .groupby([first_attr, second_attr])\
            .size()\
            .reset_index(name="count")

        return dictify_df(tmp_df)

    # noinspection PyMethodMayBeStatic
    def get_stats_pair_w_nulls(self, first_attr, second_attr, data_df):
        tmp_df = data_df[[first_attr, second_attr]]\
            .groupby([first_attr, second_attr])\
            .size()\
            .reset_index(name="count")

        return dictify_df(tmp_df)

    def add_frequency_increments_to_stats(self):
        attrs = self.get_attributes()

        for attr in attrs:
            for val, count in self.inc_single_attr_stats_w_nulls.items():
                if val in self.single_attr_stats_w_nulls[attr]:
                    # The key 'val' already exists, so we just update the count.
                    self.single_attr_stats_w_nulls[attr][val] += count
                else:
                    # The key 'val' is new, so we insert a new dictionary for 'attr'.
                    self.single_attr_stats_w_nulls[attr].update({val: count})

        for cond_attr in attrs:
            for trg_attr in attrs:
                if cond_attr != trg_attr:
                    for cond_val, nested_dict in self.inc_pair_attr_stats_w_nulls.items():
                        for trg_val, count in nested_dict.items():
                            if cond_val in self.pair_attr_stats_w_nulls[cond_attr][trg_attr]:
                                if trg_val in self.pair_attr_stats_w_nulls[cond_attr][trg_attr][cond_val]:
                                    self.pair_attr_stats_w_nulls[cond_attr][trg_attr][cond_val][trg_val] += count
                                else:
                                    new_dict = {trg_val: count}
                                    self.pair_attr_stats_w_nulls[cond_attr][trg_attr][cond_val].update(new_dict)
                            else:
                                new_dict = {cond_val: {trg_val: count}}
                                self.pair_attr_stats_w_nulls[cond_attr][trg_attr].update(new_dict)

        stats = (self.single_attr_stats_w_nulls, self.pair_attr_stats_w_nulls)

        return stats

    def get_domain_info(self):
        """
        get_domain_info returns (number of random variables, count of distinct values across all attributes).
        """
        query = 'SELECT count(_vid_), max(domain_size) FROM %s' % AuxTables.cell_domain.name
        res = self.engine.execute_query(query)
        total_vars = int(res[0][0])
        classes = int(res[0][1])
        return total_vars, classes

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
