from string import Template

import pandas as pd

from .detector import Detector

unary_template = Template('SELECT t1._tid_ FROM "$table" as t1 WHERE $cond')

unary_template_incremental = Template('SELECT t1._tid_ FROM "$repaired" as t1 WHERE $cond ' +
                                      'UNION ' +
                                      'SELECT t1._tid_ FROM "$table" as t1 WHERE $cond')

multi_template = Template('SELECT t1._tid_ FROM "$table" as t1 WHERE $cond1 $c ' +
                          'EXISTS (SELECT t2._tid_ FROM "$table" as t2 WHERE $cond2)')

multi_template_incremental = Template('SELECT t1._tid_ FROM "$repaired" as t1 WHERE $cond1 $c ' +
                                      'EXISTS (SELECT t2._tid_ FROM "$repaired" as t2 WHERE $cond2) ' +
                                      'UNION ' +
                                      'SELECT t1._tid_ FROM "$repaired" as t1 WHERE $cond1 $c ' +
                                      'EXISTS (SELECT t2._tid_ FROM "$table" as t2 WHERE $cond2) ' +
                                      'UNION ' +
                                      'SELECT t1._tid_ FROM "$table" as t1 WHERE $cond1 $c ' +
                                      'EXISTS (SELECT t2._tid_ FROM "$table" as t2 WHERE $cond2)')


class ViolationDetector(Detector):
    """
    Error detector that spots violations of integrity constraints (mainly Denial Constraints).
    """

    def __init__(self, name='ViolationDetector'):
        super(ViolationDetector, self).__init__(name)

    def setup(self, dataset):
        self.ds = dataset

    def detect_noisy_cells(self, incremental=False, first_batch=True):
        """
        Returns a pandas.DataFrame containing all cells that violate Denial Constraints in the data previously loaded.

        :return: pandas.DataFrame with columns:
            _tid_: entity ID
            attribute: attribute violating any Denial Constraint
        """

        constraints = self.ds.constraints
        tbl = self.ds.raw_data.name
        queries = []
        attrs = []

        for c in constraints:
            q = self._to_sql(tbl, c)
            queries.append(q)
            attrs.append(c.components)

        results = self.ds.engine.execute_queries(queries)

        errors = []
        for i in range(len(attrs)):
            res = results[i]
            attr_list = attrs[i]
            tmp_df = self._gen_tid_attr_output(res, attr_list)
            errors.append(tmp_df)
        errors_df = pd.concat(errors, ignore_index=True).drop_duplicates().reset_index(drop=True)

        return errors_df

    def _to_sql(self, tbl, c):
        unary = len(c.tuple_names) == 1

        if unary:
            query = self._gen_unary_query(tbl, c)
        else:
            query = self._gen_multi_query(tbl, c)

        return query

    def _gen_unary_query(self, tbl, c):
        if self.ds.incremental and not self.is_first_batch():
            query = unary_template_incremental.substitute(repaired=tbl + '_repaired',
                                                          table=tbl,
                                                          cond=c.cnf_form)
        else:
            query = unary_template.substitute(table=tbl,
                                              cond=c.cnf_form)

        return query

    def _gen_multi_query(self, tbl, c):
        cond1_preds = []
        cond2_preds = []

        for pred in c.predicates:
            if 't1' in pred.cnf_form:
                if 't2' in pred.cnf_form:
                    cond2_preds.append(pred.cnf_form)
                else:
                    cond1_preds.append(pred.cnf_form)
            elif 't2' in pred.cnf_form:
                cond2_preds.append(pred.cnf_form)
            else:
                raise Exception("ERROR in ViolationDetector: cannot ground multi-tuple template.")

        cond1 = " AND ".join(cond1_preds)
        cond2 = " AND ".join(cond2_preds)

        a = []
        for b in c.components:
            a.append("'" + b + "'")

        if cond1 != '':
            if self.ds.incremental and not self.is_first_batch():
                query = multi_template_incremental.substitute(repaired=tbl + '_repaired',
                                                              table=tbl,
                                                              cond1=cond1,
                                                              c='AND',
                                                              cond2=cond2)
            else:
                query = multi_template.substitute(table=tbl,
                                                  cond1=cond1,
                                                  c='AND',
                                                  cond2=cond2)
        else:
            if self.ds.incremental and not self.is_first_batch():
                query = multi_template_incremental.substitute(repaired=tbl + '_repaired',
                                                              table=tbl,
                                                              cond1=cond1,
                                                              c='',
                                                              cond2=cond2)
            else:
                query = multi_template.substitute(table=tbl,
                                                  cond1=cond1,
                                                  c='',
                                                  cond2=cond2)

        return query
