import logging
import time

import pandas as pd

from dataset import AuxTables


class DetectEngine:
    def __init__(self, env, dataset):
        self.env = env
        self.ds = dataset

    def detect_errors(self, detectors, batch=1):
        """
        Detects errors using a list of detectors.
        :param detectors: (list of) ErrorDetector objects
        :param batch: equal to 1 when handling initial data, greater than 1 when handling incoming data
        """
        errors = []
        tic_total = time.clock()

        # Initialize all error detectors.
        for detector in detectors:
            detector.setup(self.ds, self.env, batch)

        # Run detection using each detector.
        for detector in detectors:
            tic = time.clock()
            error_df = detector.detect_noisy_cells()
            toc = time.clock()
            logging.debug("DONE with Error Detector: %s in %.2f secs", detector.name, toc-tic)
            errors.append(error_df)

        # Get unique errors only that might have been detected from multiple detectors.
        errors_df = pd.concat(errors, ignore_index=True).drop_duplicates().reset_index(drop=True)
        errors_df['_cid_'] = errors_df.apply(lambda x: self.ds.get_cell_id(x['_tid_'], x['attribute']), axis=1)
        logging.info("detected %d potentially erroneous cells", errors_df.shape[0])

        # Store errors to db.
        if batch == 1:
            self.store_detected_errors(errors_df)
        else:
            self.store_detected_errors(errors_df, append=True)

        status = "DONE with error detection."
        toc_total = time.clock()
        detect_time = toc_total - tic_total
        return status, detect_time

    def store_detected_errors(self, errors_df, append=False):
        if errors_df.empty:
            raise Exception("ERROR: Detected errors DataFrame is empty.")

        if append:
            self.ds.generate_aux_table(AuxTables.dk_cells, errors_df, append=True)
        else:
            self.ds.generate_aux_table(AuxTables.dk_cells, errors_df, store=True)
            self.ds.aux_table[AuxTables.dk_cells].create_db_index(self.ds.engine, ['_cid_'])
