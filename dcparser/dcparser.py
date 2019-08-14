import logging
import os
import time

from .constraint import DenialConstraint


class Parser:
    """
    This class creates an interface for parsing Denial Constraints.
    """
    def __init__(self, env, dataset):
        self.env = env
        self.ds = dataset
        self.dc_strings = []
        self.dcs = []

    def load_denial_constraints(self, fpath):
        """
        Loads Denial Constraints from line-separated text file.
        
        :param fpath: filepath to text file containing Denial Constraints.
        """
        tic = time.clock()
        if not self.ds.raw_data:
            status = 'No dataset specified.'
            toc = time.clock()
            return status, toc - tic
        attrs = self.ds.raw_data.get_attributes()
        try:
            dc_file = open(fpath, 'r')
            status = "OPENED constraint file successfully."
            logging.debug(status)
            for line in dc_file:
                line = line.rstrip()
                # Skips empty and comment lines.
                if not line or line.startswith('#'):
                    continue
                self.dc_strings.append(line)
                self.dcs.append(DenialConstraint(line,attrs))
            status = 'DONE loading DCs from {fname}.'.format(fname=os.path.basename(fpath))
        except Exception:
            logging.error('FAILED to load constraints from file %s.', os.path.basename(fpath))
            raise
        toc = time.clock()
        return status, toc - tic

    def get_dcs(self):
        return self.dcs
