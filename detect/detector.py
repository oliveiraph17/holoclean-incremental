from abc import ABCMeta, abstractmethod

import pandas as pd


class Detector:
    """
    This class is an abstract class for a general error detector.
    It requires every subclass to implement the 'setup' and 'detect_noisy_cells' method.
    """
    __metaclass__ = ABCMeta

    def __init__(self, name):
        """
        Construct error detection object.
        
        :param name: The name of the error detector.
        """
        self.name = name
        self.ds = None

    @abstractmethod
    def setup(self, dataset):
        raise NotImplementedError

    @abstractmethod
    def detect_noisy_cells(self):
        """
        This method creates a DataFrame which has the information (tuple index, attribute) for the 'dk_cells' table.

        :return DataFrame for the 'dk_cells' table.
        """
        raise NotImplementedError

    # This method can be used when the Detector is able to handle incremental data.
    def is_first_batch(self):
        # Since we perform error detection before collecting statistics from the database,
        # the total number of tuples will be 0 at this point if this is the first batch of data.
        if self.ds.get_total_tuples() == 0:
            return True
        else:
            return False

    @staticmethod
    def _gen_tid_attr_output(res, attr_list):
        errors = []

        for t in res:
            tid = int(t[0])

            for attr in attr_list:
                errors.append({'_tid_': tid, 'attribute': attr})

        error_df = pd.DataFrame(data=errors)

        return error_df
