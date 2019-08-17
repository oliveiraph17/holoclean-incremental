from collections import namedtuple
import logging
import os
from string import Template
import time

import pandas as pd

from dataset import AuxTables
from dataset.table import Table, Source
from utils import NULL_REPR

EvalReport = namedtuple('EvalReport',
                        ['precision', 'recall', 'repair_recall',
                         'f1', 'repair_f1', 'detected_errors', 'total_errors', 'correct_repairs',
                         'total_repairs',
                         'total_repairs_grdt', 'total_repairs_grdt_correct', 'total_repairs_grdt_incorrect'])

errors_template = Template('SELECT COUNT(*) '
                           '  FROM $init_table AS t1, $grdt_table AS t2 '
                           ' WHERE t1._tid_ = t2._tid_ '
                           '   AND t2._attribute_ = \'$attr\' '
                           '   AND t1."$attr" != t2._value_')

errors_template_previous = Template('SELECT COUNT(*) FROM ( '
                                    '  SELECT t1._tid_ '
                                    '    FROM $init_table AS t1, $grdt_table AS t2 '
                                    '   WHERE t1._tid_ = t2._tid_ '
                                    '     AND t2._attribute_ = \'$attr\' '
                                    '     AND t1."$attr" != t2._value_ '
                                    '   UNION '
                                    '  SELECT t3._tid_ '
                                    '    FROM $rep_table_copy AS t3, $grdt_table AS t2 '
                                    '   WHERE t3._tid_ = t2._tid_ '
                                    '     AND t2._attribute_ = \'$attr\' '
                                    '     AND t3."$attr" != t2._value_ '
                                    ') AS q')

"""
The 'errors' aliased subquery returns the (_tid_, _attribute_, _value_)
from the ground-truth table for all cells that have an error in the original raw data.

The 'repairs' aliased table contains the cells and values we have inferred.

We then count the number of repaired cells whose value equals the correct ground-truth value.
"""
correct_repairs_template = Template('SELECT COUNT(*) FROM '
                                    '  (SELECT t2._tid_, t2._attribute_, t2._value_ '
                                    '     FROM $init_table as t1, $grdt_table as t2 '
                                    '    WHERE t1._tid_ = t2._tid_ '
                                    '      AND t2._attribute_ = \'$attr\' '
                                    '      AND t1."$attr" != t2._value_) AS errors, $inf_dom AS repairs '
                                    ' WHERE errors._tid_ = repairs._tid_ '
                                    '   AND errors._attribute_ = repairs.attribute '
                                    '   AND errors._value_ = repairs.rv_value')

# The template below is similar to the previous one, but considers errors
# both from the repaired and from the incoming data.
correct_repairs_template_previous = Template('SELECT COUNT(*) FROM '
                                             '  (SELECT t2._tid_, t2._attribute_, t2._value_ '
                                             '     FROM $init_table as t1, $grdt_table as t2 '
                                             '    WHERE t1._tid_ = t2._tid_ '
                                             '      AND t2._attribute_ = \'$attr\' '
                                             '      AND t1."$attr" != t2._value_ '
                                             '    UNION '
                                             '   SELECT t2._tid_, t2._attribute_, t2._value_ '
                                             '     FROM $rep_table_copy as t3, $grdt_table as t2 '
                                             '    WHERE t3._tid_ = t2._tid_ '
                                             '      AND t2._attribute_ = \'$attr\' '
                                             '      AND t3."$attr" != t2._value_) AS errors, $inf_dom AS repairs '
                                             ' WHERE errors._tid_ = repairs._tid_ '
                                             '   AND errors._attribute_ = repairs.attribute '
                                             '   AND errors._value_ = repairs.rv_value')


class EvalEngine:
    def __init__(self, env, dataset):
        self.env = env
        self.ds = dataset
        self.clean_data = None

        self.detected_errors = None
        self.total_errors = None
        self.correct_repairs = None
        self.total_repairs = None
        self.total_repairs_grdt = None
        self.total_repairs_grdt_correct = None
        self.total_repairs_grdt_incorrect = None

    def load_data(self, name, fpath, tid_col, attr_col, val_col, na_values=None):
        tic = time.clock()
        try:
            raw_data = pd.read_csv(fpath, na_values=na_values, encoding='utf-8')

            # We drop any ground-truth values that are NULLs since we follow
            # the closed-world assumption that "if it is not there, then it is wrong".
            # TODO: Revisit this once we allow users to specify which attributes may be NULL.
            raw_data.dropna(subset=[val_col], inplace=True)

            # PH: The line below seems to be useless in this context.
            raw_data.fillna(NULL_REPR, inplace=True)

            raw_data.rename({tid_col: '_tid_', attr_col: '_attribute_', val_col: '_value_'},
                            axis='columns',
                            inplace=True)

            raw_data = raw_data[['_tid_', '_attribute_', '_value_']]

            # Removes leading and trailing spaces, as well as converts the values to lowercase.
            raw_data['_value_'] = raw_data['_value_'].str.strip().str.lower()

            self.clean_data = Table(name, Source.DF, df=raw_data)
            self.clean_data.store_to_db(self.ds.engine.engine)
            self.clean_data.create_db_index(self.ds.engine, ['_tid_'])
            self.clean_data.create_db_index(self.ds.engine, ['_attribute_'])

            status = 'DONE loading {fname}.'.format(fname=os.path.basename(fpath))
        except Exception:
            logging.error('ERROR loading clean data table %s.', name)
            raise
        toc = time.clock()
        load_time = toc - tic
        return status, load_time

    def evaluate_repairs(self):
        self.compute_detected_errors()
        self.compute_total_errors()
        self.compute_correct_repairs()
        self.compute_total_repairs()
        self.compute_total_repairs_grdt()

        prec = self.compute_precision()
        rec = self.compute_recall()
        rep_recall = self.compute_repairing_recall()
        f1 = self.compute_f1()
        rep_f1 = self.compute_repairing_f1()

        if self.env['verbose']:
            self.log_weak_label_stats()

        return prec, rec, rep_recall, f1, rep_f1

    def eval_report(self):
        """
        Returns an EvalReport namedtuple containing the experimental results.
        """
        tic = time.clock()

        try:
            prec, rec, rep_recall, f1, rep_f1 = self.evaluate_repairs()

            report = "PRECISION = %.2f, RECALL = %.2f, REPAIRING RECALL = %.2f, " \
                     "F1 = %.2f, REPAIRING F1 = %.2f, DETECTED ERRORS = %d, TOTAL ERRORS = %d, " \
                     "CORRECT REPAIRS = %d, TOTAL REPAIRS = %d, " \
                     "TOTAL REPAIRS on correct cells (ground-truth present) = %d, " \
                     "TOTAL REPAIRS on incorrect cells (ground-truth present) = %d" % (
                      prec, rec, rep_recall,
                      f1, rep_f1, self.detected_errors, self.total_errors,
                      self.correct_repairs, self.total_repairs,
                      self.total_repairs_grdt_correct, self.total_repairs_grdt_incorrect)

            eval_report = EvalReport(prec, rec, rep_recall,
                                     f1, rep_f1, self.detected_errors, self.total_errors,
                                     self.correct_repairs, self.total_repairs, self.total_repairs_grdt,
                                     self.total_repairs_grdt_correct, self.total_repairs_grdt_incorrect)
        except Exception as e:
            logging.error("ERROR generating evaluation report: %s." % e)
            raise

        toc = time.clock()
        report_time = toc - tic

        return report, report_time, eval_report

    def compute_detected_errors(self):
        """
        Memoizes the number of cells spotted in error detection whose values were wrong indeed.
        Requires ground-truth data.

        This value is always equal to or less than total errors (see 'compute_total_errors').
        """
        query = "SELECT COUNT(*) FROM " \
                "  (SELECT _vid_ " \
                "     FROM %s AS t1, %s AS t2, %s AS t3 " \
                "    WHERE t1._tid_ = t2._tid_ AND t1._cid_ = t3._cid_ " \
                "      AND t1.attribute = t2._attribute_ " \
                "      AND t1.init_value != t2._value_) AS t" \
                % (AuxTables.cell_domain.name, self.clean_data.name, AuxTables.dk_cells.name)
        res = self.ds.engine.execute_query(query)
        self.detected_errors = float(res[0][0])

    def compute_total_errors(self):
        """
        Memoizes the number of cells whose initial value is wrong.
        Requires ground-truth data.
        """
        queries = []
        total_errors = 0.0
        if self.ds.is_first_batch() or not self.env['repair_previous_errors']:
            for attr in self.ds.get_attributes():
                query = errors_template.substitute(init_table=self.ds.raw_data.name,
                                                   grdt_table=self.clean_data.name,
                                                   attr=attr)
                queries.append(query)
        else:
            for attr in self.ds.get_attributes():
                query = errors_template_previous.substitute(init_table=self.ds.raw_data.name,
                                                            grdt_table=self.clean_data.name,
                                                            attr=attr,
                                                            rep_table_copy=AuxTables.repaired_table_copy.name)
                queries.append(query)
        results = self.ds.engine.execute_queries(queries)
        for res in results:
            total_errors += float(res[0][0])
        self.total_errors = total_errors

    def compute_correct_repairs(self):
        """
        Memoizes the number of cells that were correctly repaired.

        This value is always equal to or less than total repairs (see 'compute_total_repairs').
        """
        queries = []
        correct_repairs = 0.0
        if self.ds.is_first_batch() or not self.env['repair_previous_errors']:
            for attr in self.ds.get_attributes():
                query = correct_repairs_template.substitute(init_table=self.ds.raw_data.name,
                                                            grdt_table=self.clean_data.name,
                                                            attr=attr,
                                                            inf_dom=AuxTables.inf_values_dom.name)
                queries.append(query)
        else:
            for attr in self.ds.get_attributes():
                query = correct_repairs_template_previous.substitute(init_table=self.ds.raw_data.name,
                                                                     grdt_table=self.clean_data.name,
                                                                     attr=attr,
                                                                     inf_dom=AuxTables.inf_values_dom.name,
                                                                     rep_table_copy=AuxTables.repaired_table_copy.name)
                queries.append(query)
        results = self.ds.engine.execute_queries(queries)
        for res in results:
            correct_repairs += float(res[0][0])
        self.correct_repairs = correct_repairs

    def compute_total_repairs(self):
        """
        Memoizes the total number of repairs, i.e. the number of inferred cells
        whose inferred value is not equal to the initial value.
        """

        query = "SELECT COUNT(*) FROM " \
                "  (SELECT _vid_ " \
                "     FROM {} AS t1, {} AS t2 " \
                "    WHERE t1._tid_ = t2._tid_ " \
                "      AND t1.attribute = t2.attribute " \
                "      AND t1.init_value != t2.rv_value) AS t".format(AuxTables.cell_domain.name,
                                                                      AuxTables.inf_values_dom.name)
        res = self.ds.engine.execute_query(query)
        self.total_repairs = float(res[0][0])

    def compute_total_repairs_grdt(self):
        """
        Memoizes the number of repairs for cells that are specified in the clean/ground-truth data.
        Otherwise, repairs are defined the same way as in 'compute_total_repairs'.

        We also distinguish between repairs on correct cells and repairs on incorrect cells
        (correct cells are cells where init_value == ground-truth value).
        """
        query = "  SELECT (t1.init_value = t3._value_) AS is_correct, COUNT(*) " \
                "    FROM {} AS t1, {} AS t2, {} AS t3 " \
                "   WHERE t1._tid_ = t2._tid_ " \
                "     AND t1.attribute = t2.attribute " \
                "     AND t1.init_value != t2.rv_value " \
                "     AND t1._tid_ = t3._tid_ " \
                "     AND t1.attribute = t3._attribute_ " \
                "GROUP BY is_correct".format(AuxTables.cell_domain.name,
                                             AuxTables.inf_values_dom.name,
                                             self.clean_data.name)
        res = self.ds.engine.execute_query(query)

        # Memoizes the number of repairs on correct cells and incorrect cells.
        # Since we do a GROUP BY, we need to check which row of the result corresponds to the correct/incorrect count.
        self.total_repairs_grdt_correct = 0
        self.total_repairs_grdt_incorrect = 0
        self.total_repairs_grdt = 0

        if not res:
            return

        if res[0][0]:
            correct_idx, incorrect_idx = 0, 1
        else:
            correct_idx, incorrect_idx = 1, 0

        if correct_idx < len(res):
            self.total_repairs_grdt_correct = float(res[correct_idx][1])
        if incorrect_idx < len(res):
            self.total_repairs_grdt_incorrect = float(res[incorrect_idx][1])

        self.total_repairs_grdt = self.total_repairs_grdt_correct + self.total_repairs_grdt_incorrect

    def compute_precision(self):
        """
        Computes precision (# of correct repairs / # of total repairs with ground-truth)
        """
        if self.total_repairs_grdt == 0:
            return 0

        return self.correct_repairs / self.total_repairs_grdt

    def compute_recall(self):
        """
        Computes recall (# of correct repairs / # of total errors).
        """
        if self.total_errors == 0:
            return 0

        return self.correct_repairs / self.total_errors

    def compute_repairing_recall(self):
        """
        Computes repairing recall (# of correct repairs / # of total detected errors).
        """
        if self.detected_errors == 0:
            return 0

        return self.correct_repairs / self.detected_errors

    def compute_f1(self):
        prec = self.compute_precision()
        rec = self.compute_recall()

        if prec + rec == 0:
            return 0

        f1 = 2 * (prec * rec) / (prec + rec)

        return f1

    def compute_repairing_f1(self):
        prec = self.compute_precision()
        rec = self.compute_repairing_recall()

        if prec + rec == 0:
            return 0

        f1 = 2 * (prec * rec) / (prec + rec)

        return f1

    def log_weak_label_stats(self):
        query = """
            SELECT
                (t3._tid_ IS NULL) AS clean,
                (t1.fixed) AS status,
                (t4._tid_ IS NOT NULL) AS inferred,
                (t1.init_value = t2._value_) AS init_eq_grdth,
                (t1.init_value = t4.rv_value) AS init_eq_infer,
                (t1.weak_label = t1.init_value) AS wl_eq_init,
                (t1.weak_label = t2._value_) AS wl_eq_grdth,
                (t1.weak_label = t4.rv_value) AS wl_eq_infer,
                (t2._value_ = t4.rv_value) AS infer_eq_grdth,
                COUNT(*) AS count
            FROM
                {cell_domain} AS t1,
                {clean_data} AS t2
                LEFT JOIN {dk_cells} AS t3 ON t2._tid_ = t3._tid_ AND t2._attribute_ = t3.attribute
                LEFT JOIN {inf_values_dom} AS t4 ON t2._tid_ = t4._tid_ AND t2._attribute_ = t4.attribute
                WHERE t1._tid_ = t2._tid_ AND t1.attribute = t2._attribute_
            GROUP BY
                clean,
                status,
                inferred,
                init_eq_grdth,
                init_eq_infer,
                wl_eq_init,
                wl_eq_grdth,
                wl_eq_infer,
                infer_eq_grdth
        """.format(cell_domain=AuxTables.cell_domain.name,
                   clean_data=self.clean_data.name,
                   dk_cells=AuxTables.dk_cells.name,
                   inf_values_dom=AuxTables.inf_values_dom.name)

        res = self.ds.engine.execute_query(query)

        df_stats = pd.DataFrame(res, columns=["is_clean", "cell_status", "is_inferred",
                                              "init = grdth", "init = inferred",
                                              "w. label = init", "w. label = grdth", "w. label = inferred",
                                              "infer = grdth", "count"])

        df_stats = df_stats.sort_values(list(df_stats.columns)).reset_index(drop=True)

        logging.debug("Weak-label statistics:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', len(df_stats))
        pd.set_option('display.max_colwidth', -1)
        logging.debug("%s", df_stats)
        pd.reset_option('display.max_columns')
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_colwidth')
