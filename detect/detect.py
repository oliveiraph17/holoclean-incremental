import logging
import time

import pandas as pd

from dataset import AuxTables


class DetectEngine:
    def __init__(self, env, dataset):
        self.env = env
        self.ds = dataset

        if env['repair_previous_errors'] and not env['incremental']:
            raise Exception('Inconsistent parameters: repair_previous_errors=%r, incremental=%r.' %
                            (env['repair_previous_errors'], env['incremental']))

    def detect_errors(self, detectors):
        """
        Detects errors using a list of detectors.
        :param detectors: (list) of ErrorDetector objects
        """
        errors = []
        tic_total = time.clock()

        # Initialize all error detectors.
        for detector in detectors:
            detector.setup(self.ds, self.env['repair_previous_errors'])

        # Run detection using each detector.
        for detector in detectors:
            tic = time.clock()
            error_df = detector.detect_noisy_cells()
            toc = time.clock()
            logging.debug("DONE with Error Detector: %s in %.2f secs", detector.name, toc-tic)
            errors.append(error_df)

        # Get unique errors only that might have been detected from multiple detectors.
        self.errors_df = pd.concat(errors, ignore_index=True).drop_duplicates().reset_index(drop=True)
        if self.errors_df.shape[0]:
            self.errors_df['_cid_'] = self.errors_df.apply(lambda x: self.ds.get_cell_id(x['_tid_'], x['attribute']), axis=1)
        logging.info("detected %d potentially erroneous cells", self.errors_df.shape[0])

        # Store errors to db.
        self.store_detected_errors(self.errors_df)

        if not self.ds.is_first_batch() and self.env['repair_previous_errors']:
            self.set_previous_dirty_rows()

        status = "DONE with error detection."
        toc_total = time.clock()
        detect_time = toc_total - tic_total
        return status, detect_time, errors_df.shape[0]

    def store_detected_errors(self, errors_df):
        if errors_df.empty:
            raise Exception("ERROR: Detected errors dataframe is empty.")
        self.ds.generate_aux_table(AuxTables.dk_cells, errors_df, store=True)
        self.ds.aux_table[AuxTables.dk_cells].create_db_index(self.ds.engine, ['_cid_'])
        self.ds._active_attributes = sorted(errors_df['attribute'].unique())

    def set_previous_dirty_rows(self):
        query = 'SELECT t1.* FROM "{}" AS t1 WHERE t1._tid_ IN ' \
                '(SELECT t2._tid_ FROM "{}" AS t2)'.format(self.ds.raw_data.name + '_repaired',
                                                           AuxTables.dk_cells.name)

        results = self.ds.engine.execute_query(query)
        df = pd.DataFrame(results, columns=results[0].keys())

        self.ds.set_previous_dirty_rows(df)
