from string import Template

import pandas as pd

from .detector import Detector
from utils import NULL_REPR

query_template = Template('SELECT t1._tid_ FROM "$table_repaired" as t1 WHERE t1."$attribute" = \'$null\'')


class NullDetector(Detector):
    """
    Error detector that treats NULL values as errors.
    """

    def __init__(self, name='NullDetector'):
        super(NullDetector, self).__init__(name)

    def setup(self, dataset, repair_previous_errors=False):
        self.ds = dataset
        self.repair_previous_errors = repair_previous_errors

    def detect_noisy_cells(self):
        """
        detect_noisy_cells returns a pandas.DataFrame containing all cells with NULL values.

        :return: pandas.DataFrame with columns:
            _tid_: entity ID.
            attribute: attribute with NULL value for this entity.
        """
        raw_data_df = self.ds.get_raw_data()
        attributes = self.ds.get_attributes()
        errors = []

        if not self.ds.is_first_batch() and self.repair_previous_errors:
            table_repaired_name = self.ds.raw_data.name + '_repaired'

            for attr in attributes:
                query = query_template.substitute(table_repaired=table_repaired_name,
                                                  attribute=attr,
                                                  null=NULL_REPR)

                # Queries the database for potential errors in cells of attribute 'attr' that were already repaired.
                results = self.ds.engine.execute_query(query)
                table_repaired_errors_df = self._gen_tid_attr_output(results, [attr])
                errors.append(table_repaired_errors_df)

                # Filters the DataFrame of incoming tuples to get NULL cells of attribute 'attr'.
                table_errors_df = raw_data_df[raw_data_df[attr] == NULL_REPR]['_tid_'].to_frame()
                table_errors_df.insert(1, "attribute", attr)
                errors.append(table_errors_df)
            errors_df = pd.concat(errors, ignore_index=True)
        else:
            for attr in attributes:
                # Filters the DataFrame of incoming tuples to get NULL cells of attribute 'attr'.
                table_errors_df = raw_data_df[raw_data_df[attr] == NULL_REPR]['_tid_'].to_frame()
                table_errors_df.insert(1, "attribute", attr)
                errors.append(table_errors_df)
            errors_df = pd.concat(errors, ignore_index=True)

        return errors_df
