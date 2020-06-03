import numpy as np
import pandas as pd
import csv

from dataset.dbengine import DBengine
from dataset import Table, Source


class DatasetPreparer:
    def __init__(self, db_user='holocleanuser', db_pwd='abcd1234', db_name='holo', db_host='localhost'):
        self.raw_data = None
        self.engine = DBengine(db_user, db_pwd, db_name, db_host)

    def load_data(self, name, fpath, entity_col=None):
        """
        load_data takes a CSV file of the initial data, adds tuple IDs (_tid_)
        to each row to uniquely identify an 'entity', and generates unique
        index numbers for each attribute/column.

        Creates a table with the user supplied 'name' parameter (e.g. 'hospital').

        :param generate_clean_file:
        :param name: (str) name to initialize dataset with.
        :param fpath: (str) filepath to CSV input file.
        :param entity_col: (str) column containing the unique identifier/ID of an entity.
                           If None, assumes every row is a unique entity.
        :param store_to_db: stores to PostgreSQL.
        :param shuffle: performs a random shuffle in the DataFrame.
        :param sort_by: list of attributes for sorting the DataFrame.
        :param ascending: determines the order for sorting the DataFrame.
        """
        self.fpath = fpath

        # Loads raw CSV file/data into a PostgreSQL table 'name' (param).
        self.raw_data = Table(name, Source.FILE, fpath=self.fpath + '.csv', drop_null_columns=False)

        # Adds _tid_ column to dataset that uniquely identifies an entity.
        # If entity_col is not supplied, we use auto-incrementing values.
        # Otherwise, we use the entity values directly as _tid_'s.
        if entity_col is None:
            self.raw_data.df.insert(0, '_tid_', range(0, len(self.raw_data.df.index)))
        else:
            # Uses entity IDs as _tid_'s directly.
            self.raw_data.df.rename({entity_col: '_tid_'}, axis='columns', inplace=True)
        self.suffix = 'wtids'

    def sort(self, sort_by, ascending=True):
        self.raw_data.df.sort_values(inplace=True, by=sort_by, ascending=ascending)
        self.suffix = 'sorted'
        for attr in sort_by:
            self.suffix += '_' + attr

    def shuffle(self):
        self.raw_data.df = self.raw_data.df.reindex(np.random.permutation(self.raw_data.df.index))
        self.suffix = 'shuffled'

    def save(self, to_csv=True, to_db=False):
        if to_csv:
            self.raw_data.df.to_csv(self.fpath + '_' + self.suffix + '.csv', index=False)

        if to_db:
            self.raw_data.store_to_db(self.engine.engine)

    def generate_clean_file(self):
        with open(self.fpath + '_clean.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['tid', 'attribute', 'correct_val'])
            for _, row in self.raw_data.df.iterrows():
                tid = row['_tid_']
                row = row.drop(labels=['_tid_'])
                for attr_name, attr_value in zip(row.index, row.values):
                    r = [tid, attr_name, '' if pd.isna(attr_value) else attr_value]
                    writer.writerow(r)

    def generate_error_file(self):
        clean_df = pd.read_csv(self.fpath + '_clean.csv')
        errors_df_rows = []
        count = 0
        for i in range(len(self.raw_data.df.index)):
            for j in range(len(self.raw_data.df.columns) - 1):
                # Uses j+1 to skip the tid column.
                if str(self.raw_data.df.iloc[i, j + 1]) != str(clean_df.iloc[count, 2]):
                    errors_dict = {'_tid_': clean_df.iloc[count, 0],
                                   'attribute': clean_df.iloc[count, 1]}
                    errors_df_rows.append(errors_dict)
                count += 1
        errors_df = pd.DataFrame(errors_df_rows)
        errors_df.sort_values(by=['_tid_'], inplace=True)
        errors_df.to_csv(self.fpath + '_errors.csv', index=False)


gen = DatasetPreparer()

# Loads the input file, converts it to lowercase and adds tid in ascending order (like HoloClean does when the input
# file has no tid attribute).
gen.load_data('hospital', '../testdata/hospital/hospital')

# Does the same of the method above, but when the input file has a tid attribute.
# gen.load_data('hospital', '../testdata/hospital/hospital_with_tids', entity_col='_tid_')

# Shuffles the rows keeping the tids (added to the rows of input file or read from it).
# gen.shuffle()

# Sorts the rows according to the provided list and direction (ascending/descending) keeping the tids.
# gen.sort(sort_by=['MeasureName', 'Condition'], ascending=True)

# Saves the file, including the _tid_ column, to CSV and/or to the DB (used for the methods above).
gen.save()

# Generates the clean file assuming that the loaded dataset has no errors.
# gen.generate_clean_file()

# Generates the perfect error file to be used in the ErrorLoaderDetector for the loaded dataset, assuming that a dirty
# dataset was loaded and that the corresponding clean file exists.
# gen.generate_error_file()
