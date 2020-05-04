import logging

import numpy as np
import ujson


class ModelGroupGenerator:

    def __init__(self, env, dataset):
        self.env = env
        self.ds = dataset
        self.model_groups = None

        self.all_active_attributes = self.ds.get_active_attributes()
        self.all_active_attributes.sort()

        corr = self.ds.get_correlations()
        self.corr = {}
        for attr_tuple in corr:
            attr1 = attr_tuple[0]
            self.corr[attr1] = self.corr.get(attr1, {})
            for corr_attr_tuple in attr_tuple[1]:
                attr2 = corr_attr_tuple[0]
                self.corr[attr1][attr2] = corr_attr_tuple[1]

    def pair_correlation_grouping(self, thresh):
        """
        Groups attributes whose bi-directional correlation is greater than or equal to the threshold.
        :param thresh:
        """
        num_active_attrs = len(self.all_active_attributes)
        candidates_corr_strength = {}
        for idx, attr1 in enumerate(self.all_active_attributes):
            for attr2 in self.all_active_attributes[idx + 1:]:
                # We compare both correlations (i.e., attr1-attr2 and attr2-attr1) because conditional entropy is not
                # symmetric.
                if self.corr[attr1][attr2] >= thresh and self.corr[attr2][attr1] >= thresh:
                    pair_corr_strength = self.corr[attr1][attr2] + self.corr[attr2][attr1]
                    candidates_corr_strength[(attr1, attr2)] = pair_corr_strength

        groups = self._join_groups(candidates_corr_strength, reverse_sort=True)
        models_to_train = self._check_group_differences(groups)
        self.model_groups = groups

        return groups, models_to_train

    def corr_similarity_grouping(self, thresh):
        """
        Groups attributes whose distance between their correlation vectors is smaller than or equal to the threshold.
        :param thresh:
        """
        # Correlation vectors. Dimensions: [#all_active_attributes][#attributes]
        corr_vectors = {}
        for attr in self.all_active_attributes:
            corr_vectors[attr] = np.array([self.corr[attr][attr2] for attr2 in self.ds.get_attributes()])

        total_attrs = len(self.ds.get_attributes())
        candidates_dist = {}
        for idx, attr1 in enumerate(self.all_active_attributes):
            for attr2 in self.all_active_attributes[idx + 1:]:
                # Computes the L2 distance between the correlation vectors and scales it to the interval [0,1].
                dist_corr_vec = np.linalg.norm(corr_vectors[attr1] - corr_vectors[attr2], ord=2) / total_attrs
                if dist_corr_vec <= thresh:
                    candidates_dist[(attr1, attr2)] = dist_corr_vec

        groups = self._join_groups(candidates_dist, reverse_sort=False)
        models_to_train = self._check_group_differences(groups)
        self.model_groups = groups

        return groups, models_to_train

    def _join_groups(self, candidates, reverse_sort):
        """
        Auxiliar function to join groups based on the calling strategy. We scan the candidates sorted to guarantee that
        the attributes most correlation-similar or those which have the highest bi-directional correlation are grouped
        together if we need to decide between two attributes to be in a group as neither distance nor pair-correlation
        is transitive.
        :param candidates:
        :param reverse_sort:
        :return:
        """
        groups = {attr: [attr] for attr in self.all_active_attributes}
        for (attr1, attr2), dist in sorted(candidates.items(), key=lambda x: x[1], reverse=reverse_sort):
            # attr1, attr2 = candidates[dist]
            attr1_repr = self.ds.get_attr_group(attr1, groups)
            attr2_repr = self.ds.get_attr_group(attr2, groups)
            attr1_list = groups[attr1_repr]
            attr2_list = groups[attr2_repr]
            joinable = True
            for at1 in attr1_list:
                for at2 in attr2_list:
                    # Ensures that the first attribute in the key comes first lexicographically as the dict expects.
                    key = (at1, at2) if at1 < at2 else (at2, at1)
                    if key not in candidates.keys():
                        # If key is not in candidates, then dist(at1, at2) > thresh or corr(at1, at2) < thresh. So, the
                        # groups cannot be joined.
                        joinable = False
                        break
                    if not joinable:
                        break
            if joinable:
                # We sort to always set the first in lexicographical order to be the group representative.
                concat_attrs = sorted(groups[attr1_repr] + groups[attr2_repr])
                new_repr = concat_attrs[0]
                del groups[attr1_repr]
                del groups[attr2_repr]
                groups[new_repr] = concat_attrs

        return groups

    def _check_group_differences(self, groups):
        """
        Compares the new grouping to the old one and returns a list of attributes that require training.
        :param groups:
        :return:
        """
        active_groups = list(set([self.ds.get_attr_group(attr, groups) for attr in self.ds.get_active_attributes()]))
        if self.model_groups is None:
            models_to_train = active_groups
        else:
            models_to_train = []
            for new_attr in active_groups:
                new_attr_list = groups[new_attr]
                old_attr_list = self.model_groups.get(new_attr)
                if not old_attr_list or sorted(new_attr_list) != sorted(old_attr_list):
                    # We have a new group that requires training.
                    models_to_train.append(new_attr)

        return models_to_train

    def load_groups(self, base_path='/tmp/'):
        with open(base_path + self.ds.raw_data.name + '_model_groups.ujson',
                  encoding='utf-8') as f:
            self.model_groups = ujson.load(f)

        # We need to know all the active attributes (i.e., from the current batch or previous batches) to avoid
        # "forgetting" the models involving past active attributes that eventually are not active in the current batch.
        past_active_attributes = [attr for attr_list in self.model_groups.values() for attr in attr_list]
        for attr in past_active_attributes:
            if attr not in self.all_active_attributes:
                self.all_active_attributes.append(attr)
        self.all_active_attributes.sort()

    def save_groups(self, groups=None, base_path='/tmp/'):
        if groups is None:
            groups = self.model_groups
        with open(base_path + self.ds.raw_data.name + '_model_groups.ujson', 'w',
                  encoding='utf-8') as f:
            ujson.dump(groups, f, ensure_ascii=False)
