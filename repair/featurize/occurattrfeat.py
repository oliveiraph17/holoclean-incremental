import torch
from tqdm import tqdm

import pandas as pd

from .featurizer import Featurizer
from dataset import AuxTables
from utils import NULL_REPR


class OccurAttrFeaturizer(Featurizer):
    def __init__(self):
        self.all_attrs = None
        self.attrs_number = None
        self.raw_data_dict = None
        self.total = None
        self.single_stats = None
        self.pair_stats = None

        Featurizer.__init__(self)

    def specific_setup(self):
        self.name = 'OccurAttrFeaturizer'
        if not self.setup_done:
            raise Exception('Featurizer {} is not properly setup.'.format(self.name))
        self.all_attrs = self.ds.get_attributes()
        self.attrs_number = len(self.ds.attr_to_idx)
        self.raw_data_dict = {}
        self.total = None
        self.single_stats = None
        self.pair_stats = None
        self.setup_stats()

    def setup_stats(self):
        if not self.repair_previous_errors or self.ds.is_first_batch():
            self.raw_data_dict = self.ds.get_raw_data().set_index('_tid_').to_dict('index')
        else:
            df = pd.concat([self.ds.get_previous_dirty_rows(), self.ds.get_raw_data()])
            df.reset_index(drop=True, inplace=True)
            self.raw_data_dict = df.set_index('_tid_').to_dict('index')

        total, single_stats, pair_stats = self.ds.get_statistics()
        self.total = float(total)
        self.single_stats = single_stats
        self.pair_stats = pair_stats

    # noinspection PyShadowingBuiltins,PyTypeChecker
    def create_tensor(self):
        # Iterate over tuples in domain.
        tensors = []
        t = self.ds.aux_table[AuxTables.cell_domain]
        sorted_domain = t.df.reset_index().sort_values(by=['_vid_'])[['_tid_', 'attribute', '_vid_', 'domain']]
        records = sorted_domain.to_records()
        for row in tqdm(list(records)):
            # Get tuple from raw_dataset.
            tid = row['_tid_']
            tuple = self.raw_data_dict[tid]
            feat_tensor = self.gen_feat_tensor(row, tuple)
            tensors.append(feat_tensor)
        combined = torch.cat(tensors)
        return combined

    # noinspection PyShadowingBuiltins,PyUnresolvedReferences
    def gen_feat_tensor(self, row, tuple):
        tensor = torch.zeros(1, self.classes, self.attrs_number * self.attrs_number)

        rv_attr = row['attribute']
        domain = row['domain'].split('|||')
        rv_domain_idx = {val: idx for idx, val in enumerate(domain)}

        # We should not have any NULLs in our domain.
        assert NULL_REPR not in rv_domain_idx

        rv_attr_idx = self.ds.attr_to_idx[rv_attr]

        for attr in self.all_attrs:
            val = tuple[attr]

            # Ignore co-occurrence with the same attribute, when the value is NULL,
            # as well as when the value only co-occurs with NULL values.
            if attr == rv_attr \
                    or val == NULL_REPR \
                    or self.pair_stats[attr][rv_attr][val] == [NULL_REPR]:
                continue

            attr_idx = self.ds.attr_to_idx[attr]
            count1 = float(self.single_stats[attr][val])
            all_vals = self.pair_stats[attr][rv_attr][val]

            for rv_val in domain:
                count2 = float(all_vals.get(rv_val, 0.0))
                prob = count2 / count1

                if rv_val in rv_domain_idx:
                    index = rv_attr_idx * self.attrs_number + attr_idx
                    tensor[0][rv_domain_idx[rv_val]][index] = prob

        return tensor

    def feature_names(self):
        return ["{} X {}".format(attr1, attr2) for attr1 in self.all_attrs for attr2 in self.all_attrs]
