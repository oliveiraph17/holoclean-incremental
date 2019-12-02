from abc import ABCMeta, abstractmethod

import pandas as pd


class Detector:
    """
    This class is an abstract class for general error detection,
     it requires for every sub-class to implement the
    setup and detect_noisy_cells method
    """
    __metaclass__ = ABCMeta

    def __init__(self, name):
        """
        Construct error detection object
        
        :param name: The name of the error detector
        """
        self.name = name
        self.ds = None

    @abstractmethod
    def setup(self, dataset, env):
        raise NotImplementedError

    @abstractmethod
    def detect_noisy_cells(self):
        """
        This method creates a dataframe which has the information
        (tuple index,attribute) for the dk_cells

        :return dataframe  for the dk_cell
        """
        raise NotImplementedError

    @staticmethod
    def _gen_tid_attr_output(res, attr_list):
        errors = []

        for t in res:
            tid = int(t[0])

            for attr in attr_list:
                errors.append({'_tid_': tid, 'attribute': attr})

        error_df = pd.DataFrame(data=errors)

        return error_df

    @staticmethod
    def create_empty_errors_df():
        empty_errors_df = pd.DataFrame(columns=['_tid_', 'attribute', '_cid_'])
        empty_errors_df['_tid_'] = empty_errors_df['_tid_'].astype(int)
        empty_errors_df['attribute'] = empty_errors_df['attribute'].astype(str)
        empty_errors_df['_cid_'] = empty_errors_df['_cid_'].astype(int)

        return empty_errors_df
