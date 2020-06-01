import logging
import time

import pandas as pd

from dataset import AuxTables


class DetectEngine:
    def __init__(self, env, dataset):
        self.env = env
        self.ds = dataset
        self.detectors = None

        if env['repair_previous_errors'] and not env['incremental']:
            raise Exception('Inconsistent parameters: repair_previous_errors=%r, incremental=%r.' %
                            (env['repair_previous_errors'], env['incremental']))

    def detect_errors(self, detectors):
        """
        Detects errors using a list of detectors.
        :param detectors: (list) of ErrorDetector objects
        """
        errors = []
        errors_all_batches = None

        # We identify errors in cells from previous batches to not include them in the training set when using the
        # combination of options train_using_all_batches=True and repair_previous_errors=False.
        if not self.ds.is_first_batch() and self.env['train_using_all_batches'] \
                and not self.env['repair_previous_errors']:
            errors_all_batches = []

        self.detectors = detectors
        tic_total = time.clock()

        # Initialize all error detectors.
        for detector in self.detectors:
            detector.setup(self.ds, self.env)

        # Run detection using each detector.
        for detector in self.detectors:
            tic = time.clock()
            error_df = detector.detect_noisy_cells()
            toc = time.clock()
            logging.debug("DONE with Error Detector: %s in %.2f secs", detector.name, toc-tic)
            errors.append(error_df)

            if errors_all_batches is not None:
                self.env['repair_previous_errors'] = True
                error_all_batches_df = detector.detect_noisy_cells()
                errors_all_batches.append(error_all_batches_df)
                self.env['repair_previous_errors'] = False

        # Get unique errors only that might have been detected from multiple detectors.
        self.errors_df = pd.concat(errors, ignore_index=True).drop_duplicates().reset_index(drop=True)
        if self.errors_df.shape[0]:
            found_errors = True
            self.errors_df['_cid_'] = self.errors_df.apply(lambda x: self.ds.get_cell_id(x['_tid_'], x['attribute']),
                                                           axis=1)
        else:
            found_errors = False
        logging.info("detected %d potentially erroneous cells", self.errors_df.shape[0])

        # Store errors to db.
        self.store_detected_errors(self.errors_df)

        if found_errors and not self.ds.is_first_batch() and self.env['repair_previous_errors']:
            self.set_previous_dirty_rows()

        # Do the same for errors from previous batches to not include them in the training set.
        if errors_all_batches is not None:
            errors_all_batches_df = pd.concat(
                errors_all_batches, ignore_index=True).drop_duplicates().reset_index(drop=True)
            if errors_all_batches_df.shape[0]:
                errors_all_batches_df['_cid_'] = errors_all_batches_df.apply(
                    lambda x: self.ds.get_cell_id(x['_tid_'], x['attribute']), axis=1)
            self.store_detected_errors_all_batches(errors_all_batches_df)

        status = "DONE with error detection."
        toc_total = time.clock()
        detect_time = toc_total - tic_total

        return status, detect_time, self.errors_df.shape[0], found_errors

    def store_detected_errors(self, errors_df):
        if errors_df.empty:
            logging.info("Detected errors dataframe is empty.")
        self.ds.generate_aux_table(AuxTables.dk_cells, errors_df, store=True)
        self.ds.aux_table[AuxTables.dk_cells].create_db_index(self.ds.engine, ['_cid_'])
        self.ds._active_attributes = sorted(errors_df['attribute'].unique())

    def store_detected_errors_all_batches(self, errors_all_batches_df):
        if errors_all_batches_df.empty:
            logging.info("Detected errors all batches dataframe is empty.")
        self.ds.generate_aux_table(AuxTables.dk_cells_all_batches, errors_all_batches_df, store=True)
        self.ds.aux_table[AuxTables.dk_cells_all_batches].create_db_index(self.ds.engine, ['_cid_'])

    def set_previous_dirty_rows(self):
        query = 'SELECT t1.* FROM "{}" AS t1 WHERE t1._tid_ IN ' \
                '(SELECT t2._tid_ FROM "{}" AS t2)'.format(self.ds.raw_data.name + '_repaired',
                                                           AuxTables.dk_cells.name)

        results = self.ds.engine.execute_query(query)

        if results:
            df = pd.DataFrame(results, columns=results[0].keys())
        else:
            df = pd.DataFrame(results, columns=self.ds.raw_data.get_attributes())

        self.ds.set_previous_dirty_rows(df)
