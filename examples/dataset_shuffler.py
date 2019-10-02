import numpy as np
import pandas as pd

from dataset.dbengine import DBengine
from dataset import Table, Source

class DatasetShuffler:

    def __init__(self, db_user='holocleanuser', db_pwd='abcd1234', db_name='holo', db_host='localhost'):
        self.raw_data = None
        self.engine = DBengine(db_user, db_pwd, db_name, db_host)

    def load_data(self, name, fpath, entity_col=None,
                  store_to_db=False, shuffle=False, sort_by=[], ascending=True):
        """
        load_data takes a CSV file of the initial data, adds tuple IDs (_tid_)
        to each row to uniquely identify an 'entity', and generates unique
        index numbers for each attribute/column.

        Creates a table with the user supplied 'name' parameter (e.g. 'hospital').

        :param name: (str) name to initialize dataset with.
        :param fpath: (str) filepath to CSV input file.
        :param entity_col: (str) column containing the unique
            identifier/ID of an entity.  For fusion tasks, rows with
            the same ID will be fused together in the output.
            If None, assumes every row is a unique entity.
        :param store_to_db: stores to PostgreSQL
        :param shuffle: performs a random shuffle in the DataFrame
        :param sort_by: list of attributes for sorting the DataFrame
        """
        # Load raw CSV file/data into a Postgres table 'name' (param).
        self.raw_data = Table(name, Source.FILE, fpath=fpath + '.csv')

        # Add _tid_ column to dataset that uniquely identifies an entity.
        # If entity_col is not supplied, use auto-incrementing values.
        # Otherwise we use the entity values directly as _tid_'s.
        if entity_col is None:
            self.raw_data.df.insert(0, '_tid_', range(0, len(self.raw_data.df.index)))
        else:
            # use entity IDs as _tid_'s directly
            self.raw_data.df.rename({entity_col: '_tid_'}, axis='columns', inplace=True)

        # Rearrange the dataset according to the input parameters
        df = self.raw_data.df.copy()
        suffix = 'wtids'
        if shuffle:
            self.raw_data.df = df.reindex(np.random.permutation(df.index))
            suffix = 'shuffled'
        elif not sort_by == []:
            self.raw_data.df.sort_values(inplace=True, by=sort_by, ascending=ascending)
            suffix = 'sorted'
            for attr in sort_by:
                suffix += '_' + attr

        self.raw_data.df.to_csv(fpath + '_' + suffix + '.csv', index=False)

        if store_to_db:
            self.raw_data.store_to_db(self.engine.engine)

gen = DatasetShuffler()

# Generates a new CSV with tid set ascending (like HoloClean does when the input file has no tid attribute)
# gen.load_data('hospital', '../testdata/hospital/hospital')

# Generates a new CSV with the same tids of the input file but shuffled
gen.load_data('hospital', '../testdata/hospital/hospital', shuffle=True)

# Generates a new CSV with the same tids of the input file but sorted according to the provided list
# gen.load_data('hospital', '../testdata/hospital/hospital', sort_by=['City', 'ZipCode'])

# Use this pattern when the input file has a tid attribute
# gen.load_data('hospital', '../testdata/hospital/hospital_with_tids', entity_col='_tid_', shuffle=True)