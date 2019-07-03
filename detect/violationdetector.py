from string import Template

import pandas as pd

from .detector import Detector

unary_template = Template('SELECT t1._tid_ FROM "$table" as t1 WHERE $batch $cond')

multi_template = Template('SELECT t1._tid_ FROM "$table" as t1 WHERE $cond1 $c ' +
                          'EXISTS (SELECT t2._tid_ FROM "$table" as t2 WHERE $cond2)')

multi_template_inc = Template('SELECT t1._tid_ FROM "$table" as t1 WHERE $batch $cond1 $c ' +
                              'EXISTS (SELECT t2._tid_ FROM "$table" as t2 WHERE $batch $cond2) ' +
                              'UNION ' +
                              'SELECT t1._tid_ FROM "$table" as t1 WHERE $batch $cond1 $c ' +
                              'EXISTS (SELECT t2._tid_ FROM "$repaired" as t2 WHERE $cond2)')


class ViolationDetector(Detector):
    """
    Error detector that spots violations of integrity constraints (mainly Denial Constraints).
    """

    def __init__(self, name='ViolationDetector'):
        super(ViolationDetector, self).__init__(name)
        self.constraints = None
        self.batch_number = None
        self.batch_cond = None

    def setup(self, dataset, env, batch=1):
        self.ds = dataset
        self.env = env
        self.constraints = dataset.constraints
        self.batch_number = batch
        self.batch_cond = '_batch_ = ' + str(batch) + ' AND '

    def detect_noisy_cells(self):
        """
        Returns a pandas.DataFrame containing all cells that violate Denial Constraints in the data previously loaded.

        :return: pandas.DataFrame with columns:
            _tid_: entity ID
            attribute: attribute violating any Denial Constraint
        """

        tbl = self.ds.raw_data.name
        queries = []
        attrs = []

        for c in self.constraints:
            q = self.to_sql(tbl, c)
            queries.append(q)
            attrs.append(c.components)

        results = self.ds.engine.execute_queries(queries)

        errors = []
        for i in range(len(attrs)):
            res = results[i]
            attr_list = attrs[i]
            tmp_df = self.gen_tid_attr_output(res, attr_list)
            errors.append(tmp_df)
        errors_df = pd.concat(errors, ignore_index=True).drop_duplicates().reset_index(drop=True)

        return errors_df

    def to_sql(self, tbl, c):
        unary = len(c.tuple_names) == 1

        if unary:
            query = self.gen_unary_query(tbl, c)
        else:
            query = self.gen_multi_query(tbl, c)

        return query

    def gen_unary_query(self, tbl, c):
        query = unary_template.substitute(table=tbl,
                                          batch=self.batch_cond,
                                          cond=c.cnf_form)

        return query

    def gen_multi_query(self, tbl, c):
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
                raise Exception("ERROR in Violation Detector. Cannot ground multi-tuple template.")

        cond1 = " AND ".join(cond1_preds)
        cond2 = " AND ".join(cond2_preds)

        a = []
        for b in c.components:
            a.append("'" + b + "'")

        if cond1 != '':
            if self.batch_number == 1:
                query = multi_template.substitute(table=tbl,
                                                  cond1=cond1,
                                                  c='AND',
                                                  cond2=cond2)
            else:
                query = multi_template_inc.substitute(table=tbl,
                                                      batch=self.batch_cond,
                                                      cond1=cond1,
                                                      c='AND',
                                                      cond2=cond2,
                                                      repaired=tbl + '_repaired')
        else:
            if self.batch_number == 1:
                query = multi_template.substitute(table=tbl,
                                                  cond1=cond1,
                                                  c='',
                                                  cond2=cond2)
            else:
                query = multi_template_inc.substitute(table=tbl,
                                                      batch=self.batch_cond,
                                                      cond1=cond1,
                                                      c='',
                                                      cond2=cond2,
                                                      repaired=tbl + '_repaired')

        return query

    # noinspection PyMethodMayBeStatic
    def gen_tid_attr_output(self, res, attr_list):
        errors = []

        for t in res:
            tid = int(t[0])

            for attr in attr_list:
                errors.append({'_tid_': tid, 'attribute': attr})

        error_df = pd.DataFrame(data=errors)

        return error_df
