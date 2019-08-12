from enum import Enum
import logging

import pandas as pd


class Source(Enum):
    FILE = 1
    DF = 2
    DB = 3
    SQL = 4


class Table:
    """
    A wrapper class for dataset tables.
    """

    # noinspection PyDefaultArgument
    def __init__(self, name, src, na_values=None, exclude_attr_cols=['_tid_'],
                 fpath=None, df=None, schema_name=None, table_query=None, db_engine=None):
        """
        :param name: (str) name to assign to dataset.
        :param na_values: (str or list[str]) values to interpret as NULL.
        :param exclude_attr_cols: (list[str]) list of columns NOT to treat as attributes during training/learning.
        :param src: (Source) type of source to load from.
            Note additional parameters MUST be provided for each specific source.
                Source.FILE: :param fpath:, to read from CSV file.
                Source.DF: :param df:, to read from pandas DataFrame.
                Source.DB: :param db_engine:, to read from database table with :param name:.
                Source.SQL: :param table_query: and :param db_engine:, to use result from :param table_query:.
        :param fpath: (str) file path to CSV file containing raw data.
        :param df: (pandas.DataFrame) DataFrame containing the raw ingested data.
        :param schema_name: (str) schema used while loading Source.DB.
        :param table_query: (str) SQL query to build table from.
        :param db_engine: (DBEngine) database engine object.
        """
        self.name = name
        self.index_count = 0
        # Copies the list to memoize it.
        self.exclude_attr_cols = list(exclude_attr_cols)
        self.df = pd.DataFrame()

        if src == Source.FILE:
            if fpath is None:
                raise Exception("ERROR while loading table."
                                "File path for CSV file name expected."
                                "Please provide <fpath> param.")

            # TODO(richardwu): use COPY FROM instead of loading this into memory.
            # TODO(richardwu): No support for numerical values. To be added.
            self.df = pd.read_csv(fpath, dtype=str, na_values=na_values, encoding='utf-8')

            # Normalizes the DataFrame:
            # drops NULL columns, converts data into lowercase strings, and strips whitespaces.
            for attr in self.df.columns.values:
                if self.df[attr].isnull().all():
                    logging.warning("Dropping the following NULL column from the dataset: '%s'", attr)
                    self.df.drop(labels=[attr], axis=1, inplace=True)
                    continue
                if attr not in exclude_attr_cols:
                    self.df[attr] = self.df[attr].str.strip().str.lower()
        elif src == Source.DF:
            if df is None:
                raise Exception("ERROR while loading table."
                                "DataFrame expected."
                                "Please provide <df> param.")
            self.df = df
        elif src == Source.DB:
            if db_engine is None:
                raise Exception("ERROR while loading table."
                                "Database connection expected."
                                "Please provide <db_engine>.")
            self.df = pd.read_sql_table(name, db_engine.conn, schema=schema_name)
        elif src == Source.SQL:
            if table_query is None or db_engine is None:
                raise Exception("ERROR while loading table."
                                "SQL query and database connection expected."
                                "Please provide <table_query> and <db_engine>.")
            db_engine.create_db_table_from_query(self.name, table_query)
            self.df = pd.read_sql_table(name, db_engine.conn)

    def store_to_db(self, db_conn, if_exists='replace', index=False, index_label=None):
        # TODO: This version supports single session, single worker.
        self.df.to_sql(self.name, db_conn, if_exists=if_exists, index=index, index_label=index_label)

    def get_attributes(self):
        """
        Returns the trainable/learnable attributes, excluding meta-columns such as '_tid_'.
        """
        if self.df.empty:
            raise Exception("Empty DataFrame for table {name}. Cannot return attributes.".format(name=self.name))
        return list(col for col in self.df.columns if col not in self.exclude_attr_cols)

    def create_df_index(self, attr_list):
        self.df.set_index(attr_list, inplace=True)

    def create_db_index(self, db_engine, attr_list):
        index_name = '{name}_{idx}'.format(name=self.name, idx=self.index_count)
        db_engine.create_db_index(index_name, self.name, attr_list)
        self.index_count += 1
