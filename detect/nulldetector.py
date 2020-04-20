from string import Template

import pandas as pd

from .detector import Detector
from utils import NULL_REPR

query_template = Template('SELECT t1._tid_ FROM "$table_repaired" as t1 WHERE t1."$attribute" = \'$null\'')
query_template_num = Template('SELECT t1._tid_ FROM "$table_repaired" as t1 WHERE t1."$attribute" IS NULL')


class NullDetector(Detector):
    """
    An error detector that treats null values as errors.
    """

    def __init__(self, name='NullDetector'):
        super(NullDetector, self).__init__(name)

    def setup(self, dataset, env):
        self.ds = dataset
        self.env = env
        self.df = self.ds.get_raw_data()

    def detect_noisy_cells(self):
        """
        detect_noisy_cells returns a pandas.DataFrame containing all cells with
        NULL values.

        :return: pandas.DataFrame with columns:
            _tid_: entity ID
            attribute: attribute with NULL value for this entity
        """
        errors = []
        for attr in self.ds.get_attributes():
            tmp_df = self.df[self.df[attr] == NULL_REPR]['_tid_'].to_frame()
            tmp_df.insert(1, "attribute", attr)
            errors.append(tmp_df)

        if not self.ds.is_first_batch() and self.env['repair_previous_errors']:
            # Queries the database for potential errors in cells from previous batches.
            table_repaired_name = self.ds.raw_data.name + '_repaired'

            for attr in self.ds.get_attributes():
                # if attr in self.ds.numerical_attrs:
                #     query = query_template_num.substitute(table_repaired=table_repaired_name,
                #                                           attribute=attr)
                # else:
                query = query_template.substitute(table_repaired=table_repaired_name,
                                                  attribute=attr,
                                                  null=NULL_REPR)

                results = self.ds.engine.execute_query(query)
                tmp_df = self._gen_tid_attr_output(results, [attr])
                errors.append(tmp_df)

        errors_df = pd.concat(errors, ignore_index=True)
        return errors_df
