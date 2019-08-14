from string import Template
from functools import partial

import torch
import torch.nn.functional as f

from .featurizer import Featurizer
from dataset import AuxTables
from dcparser.constraint import is_symmetric

# 'unary_template' is used for constraints where the current predicate
# used for detecting violations in pos_values have a reference to only
# one relation's (e.g. t1) attribute on one side and a fixed constant
# value on the other side of the comparison.
unary_template = Template('SELECT _vid_, val_id, count(*) violations '
                          'FROM   "$init_table" as t1, $pos_values as t2 '
                          'WHERE  t1._tid_ = t2._tid_ '
                          '  AND  t2.attribute = \'$rv_attr\' '
                          '  AND  $orig_predicates '
                          '  AND  t2.rv_val $operation $rv_val '
                          'GROUP BY _vid_, val_id')

# 'binary_template' is used for constraints where the current predicate
# used for detecting violations in pos_values have a reference to both
# relations (t1, t2), i.e. no constant value in predicate.
binary_template = Template('SELECT _vid_, val_id, count(*) violations '
                           'FROM   "$init_table" as t1, "$init_table" as t2, $pos_values as t3 '
                           'WHERE  t1._tid_ != t2._tid_ '
                           '  AND  $join_rel._tid_ = t3._tid_ '
                           '  AND  t3.attribute = \'$rv_attr\' '
                           '  AND  $orig_predicates '
                           '  AND  t3.rv_val $operation $rv_val '
                           'GROUP BY _vid_, val_id')

# 'ex_binary_template' is used as a fallback for binary_template if the latter takes too long to query.
# Instead of counting the # of violations, it returns a 0-1 indicator of whether the value violates the constraint.
ex_binary_template = Template('SELECT _vid_, val_id, 1 violations '
                              'FROM   "$init_table" as $join_rel, $pos_values as t3 '
                              'WHERE  $join_rel._tid_ = t3._tid_ '
                              '  AND  t3.attribute = \'$rv_attr\' '
                              '  AND EXISTS (SELECT $other_rel._tid_ '
                              '              FROM   "$init_table" AS $other_rel '
                              '              WHERE  $join_rel._tid_ != $other_rel._tid_ '
                              '                AND  $orig_predicates '
                              '                AND  t3.rv_val $operation $rv_val)')


# noinspection PyUnresolvedReferences
def gen_feat_tensor(violations, total_vars, classes):
    tensor = torch.zeros(total_vars, classes, 1)
    if violations:
        for entry in violations:
            vid = int(entry[0])
            val_id = int(entry[1]) - 1
            feat_val = float(entry[2])
            tensor[vid][val_id][0] = feat_val
    return tensor


class ConstraintFeaturizer(Featurizer):
    def __init__(self):
        self.constraints = None
        self.init_table_name = None
        Featurizer.__init__(self)

    def specific_setup(self):
        self.name = 'ConstraintFeaturizer'
        self.constraints = self.ds.constraints
        self.init_table_name = self.ds.raw_data.name

    def create_tensor(self):
        queries = self.generate_relaxed_sql()
        results = self.ds.engine.execute_queries_w_backup(queries)
        tensors = self._apply_func(partial(gen_feat_tensor,
                                           total_vars=self.total_vars,
                                           classes=self.classes), results)
        combined = torch.cat(tensors, 2)
        combined = f.normalize(combined, p=2, dim=1)
        return combined

    def generate_relaxed_sql(self):
        query_list = []
        for c in self.constraints:
            # Checks tuples in constraint.
            unary = (len(c.tuple_names) == 1)
            if unary:
                queries = self.gen_unary_queries(c)
            else:
                queries = self.gen_binary_queries(c)
            query_list.extend(queries)
        return query_list

    def execute_queries(self, queries):
        return self.ds.engine.execute_queries_w_backup(queries)

    @staticmethod
    def relax_unary_predicate(predicate):
        """
        Returns the attribute, operation, and tuple attribute reference.

        :return: (attr, op, const), for example:
            ("StateAvg", "<>", 't1."StateAvg"')
        """
        attr = predicate.components[0][1]
        op = predicate.operation
        comp = predicate.components[1]
        # Does not quote literals/constants in comparison.
        const = comp if comp.startswith('\'') else '"{}"'.format(comp)
        return attr, op, const

    @staticmethod
    def relax_binary_predicate(predicate, rel_idx):
        """
        Returns the attribute, operation, and tuple attribute reference.

        :return: (attr, op, const), for example:
            ("StateAvg", "<>", 't1."StateAvg"')
        """
        attr = predicate.components[rel_idx][1]
        op = predicate.operation
        const = '{}."{}"'.format(
                predicate.components[1 - rel_idx][0],
                predicate.components[1 - rel_idx][1])

        return attr, op, const

    @staticmethod
    def get_binary_predicate_join_rel(predicate):
        if 't1' in predicate.cnf_form and 't2' in predicate.cnf_form:
            if is_symmetric(predicate.operation):
                return True, ['t1'], ['t2']
            else:
                return True, ['t1', 't2'], ['t2', 't1']
        elif 't1' in predicate.cnf_form and 't2' not in predicate.cnf_form:
            return False, ['t1'], None
        elif 't1' not in predicate.cnf_form and 't2' in predicate.cnf_form:
            return False, ['t2'], None

    def gen_unary_queries(self, constraint):
        # Iterates over predicates and relax one predicate at a time.
        queries = []
        predicates = constraint.predicates
        for k in range(len(predicates)):
            orig_cnf = self._orig_cnf(predicates, k)
            # If there are no other predicates in the constraint, appends TRUE to the WHERE condition.
            # This avoids having multiple SQL templates.
            if len(orig_cnf) == 0:
                orig_cnf = 'TRUE'
            rv_attr, op, rv_val = self.relax_unary_predicate(predicates[k])
            query = unary_template.substitute(init_table=self.init_table_name,
                                              pos_values=AuxTables.pos_values.name,
                                              orig_predicates=orig_cnf,
                                              rv_attr=rv_attr,
                                              operation=op,
                                              rv_val=rv_val)
            queries.append((query, ''))
        return queries

    def gen_binary_queries(self, constraint):
        queries = []
        predicates = constraint.predicates
        for k in range(len(predicates)):
            orig_cnf = self._orig_cnf(predicates, k)
            # If there are no other predicates in the constraint, appends TRUE to the WHERE condition.
            # This avoids having multiple SQL templates.
            if len(orig_cnf) == 0:
                orig_cnf = 'TRUE'
            is_binary, join_rel, other_rel = self.get_binary_predicate_join_rel(predicates[k])
            if not is_binary:
                rv_attr, op, rv_val = self.relax_unary_predicate(predicates[k])
                query = binary_template.substitute(init_table=self.init_table_name,
                                                   pos_values=AuxTables.pos_values.name,
                                                   join_rel=join_rel[0],
                                                   orig_predicates=orig_cnf,
                                                   rv_attr=rv_attr,
                                                   operation=op,
                                                   rv_val=rv_val)
                queries.append((query, ''))
            else:
                for idx, rel in enumerate(join_rel):
                    rv_attr, op, rv_val = self.relax_binary_predicate(predicates[k], idx)
                    # Counts number of queries.
                    query = binary_template.substitute(init_table=self.init_table_name,
                                                       pos_values=AuxTables.pos_values.name,
                                                       join_rel=rel,
                                                       orig_predicates=orig_cnf,
                                                       rv_attr=rv_attr,
                                                       operation=op,
                                                       rv_val=rv_val)
                    # Fallback to 0-1 indicator instead of count.
                    ex_query = ex_binary_template.substitute(init_table=self.init_table_name,
                                                             pos_values=AuxTables.pos_values.name,
                                                             join_rel=rel,
                                                             orig_predicates=orig_cnf,
                                                             rv_attr=rv_attr,
                                                             operation=op,
                                                             rv_val=rv_val,
                                                             other_rel=other_rel[idx])
                    queries.append((query, ex_query))
        return queries

    @staticmethod
    def _orig_cnf(predicates, idx):
        """
        Returns the CNF of the predicates that does not include the predicate at index :param idx:.

        This CNF is usually used for the left relation when counting violations.
        """
        orig_preds = predicates[:idx] + predicates[(idx + 1):]
        orig_cnf = " AND ".join([pred.cnf_form for pred in orig_preds])
        return orig_cnf

    def feature_names(self):
        return ["fixed pred: {}, violation pred: {}".format(self._orig_cnf(constraint.predicates, idx),
                                                            constraint.predicates[idx].cnf_form)
                for constraint in self.constraints
                for idx in range(len(constraint.predicates))]
