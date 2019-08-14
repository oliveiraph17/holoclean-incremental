from functools import partial

import torch

from dataset import AuxTables
from .featurizer import Featurizer


# noinspection PyUnresolvedReferences,PyTypeChecker,PyShadowingBuiltins
def gen_feat_tensor(input, classes, total_attrs):
    attr_idx = input[1]
    init_idx = int(input[2])
    tensor = -1 * torch.ones(1, classes, total_attrs)
    tensor[0][init_idx][attr_idx] = 1.0
    return tensor


class InitAttrFeaturizer(Featurizer):

    # noinspection PyUnresolvedReferences
    def __init__(self, init_weight=1.0):
        """
        InitAttrFeaturizer cannot be learnable.

        :param init_weight: (float or list of floats) a fixed weight for all attributes
                            or a list of floats that represent the weights of attributes
                            in the same order of the dataset.
        """
        if isinstance(init_weight, list):
            # If 'init_weight' is a list, we convert it to a tensor so it is correctly
            # initialized in the 'TiedLinear' model instantiation, where 'init_weight'
            # is multiplied by a tensor of ones.
            init_weight = torch.FloatTensor(init_weight)

        self.all_attrs = None
        self.attr_to_idx = None
        self.total_attrs = None

        Featurizer.__init__(self, learnable=False, init_weight=init_weight)

    # noinspection PyUnresolvedReferences
    def specific_setup(self):
        self.name = 'InitAttrFeaturizer'
        self.all_attrs = self.ds.get_attributes()
        self.attr_to_idx = self.ds.attr_to_idx
        self.total_attrs = len(self.ds.attr_to_idx)

        # Makes sure that the size of 'init_weight' is equal to the number of attributes in the dataset.
        if isinstance(self.init_weight, torch.FloatTensor):
            if self.init_weight.shape[0] != len(self.all_attrs):
                raise ValueError(
                    "The size of init_weight for InitAttrFeaturizer %d does not match the number of attributes %d."
                    % (self.init_weight.shape[0], len(self.all_attrs)))

    def create_tensor(self):
        query = 'SELECT _vid_, attribute, init_index FROM %s ORDER BY _vid_' % AuxTables.cell_domain.name
        results = self.ds.engine.execute_query(query)
        map_input = []
        for res in results:
            map_input.append((res[0], self.attr_to_idx[res[1]], res[2]))
        tensors = self._apply_func(partial(gen_feat_tensor,
                                           classes=self.classes,
                                           total_attrs=self.total_attrs), map_input)
        combined = torch.cat(tensors)
        return combined

    def feature_names(self):
        return self.all_attrs
