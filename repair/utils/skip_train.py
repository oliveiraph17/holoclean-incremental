import logging

import ujson
import numpy as np
from scipy.stats import entropy


class TrainingSkipper:

    def __init__(self, env, dataset, smoothing_value=0.0001):
        self.env = env
        self.ds = dataset
        self.smoothing_value = smoothing_value
        self.pair_stats = None

    def _attr_pair_stats_to_pdf(self, attr_pair_stats1, attr_pair_stats2):
        """
        Smooths only the past PDF (stats1) as we assume that new values can appear, but no one is removed from the
        distribution (i.e., a value can be in the current PDF but not in the past PDF, but not the opposite).
        :param attr_pair_stats1:
        :param attr_pair_stats2:
        :return:
        """
        # Aligns the two pair stats into lists and prepares stats1 for smoothing.
        pair_stats_list1 = []
        pair_stats_list2 = []
        smoothing_count = 0
        for val1 in sorted(attr_pair_stats2.keys()):
            for val2, freq2 in sorted(attr_pair_stats2[val1].items()):
                if val1 in attr_pair_stats1 and val2 in attr_pair_stats1[val1]:
                    pair_stats_list1.append(attr_pair_stats1[val1][val2])
                else:
                    pair_stats_list1.append(0)
                    smoothing_count += 1
                pair_stats_list2.append(freq2)

        if not (pair_stats_list1 and pair_stats_list2):
            return None, None

        # Normalizes the frequencies to get a valid data distribution (smoothed for stats1).
        pdf_pair_stats1 = np.array(pair_stats_list1)
        pdf_pair_stats2 = np.array(pair_stats_list2)
        sum_freq1 = pdf_pair_stats1.sum()
        sum_freq2 = pdf_pair_stats2.sum()

        # Avoids division by zero.
        if (pdf_pair_stats1.size - smoothing_count == 0) or sum_freq1 == 0 or sum_freq2 == 0:
            return None, None

        smoothing_correction = (self.smoothing_value * smoothing_count) / (pdf_pair_stats1.size - smoothing_count)
        pdf_pair_stats1 = np.where(pdf_pair_stats1 != 0,
                                   (pdf_pair_stats1 / sum_freq1) - smoothing_correction,
                                   self.smoothing_value)
        pdf_pair_stats2 = np.true_divide(pdf_pair_stats2, sum_freq2)

        return pdf_pair_stats1, pdf_pair_stats2

    def get_models_to_train_individual_kl(self, thresh, models_set_to_train=None, verbose=False):
        """
        Gets the lists of attributes that will be considered for training.
        This strategy selects a model to be trained if the KL indicates that any attribute in the group changed more
        than the threshold.
        :param thresh:
        :param models_set_to_train:
        :param verbose:
        :return:
        """
        groups = self.ds.get_model_groups()
        _, _, curr_pair_stats = self.ds.get_statistics()

        if self.pair_stats is None:
            logging.debug('No loaded stats to compare. Train all models.')
            # Sets self.pair_stats to the current stats to be eventually saved later.
            self.pair_stats = curr_pair_stats
            return [att for att in groups.keys()]

        if models_set_to_train is not None:
            models_to_train = models_set_to_train
        else:
            models_to_train = []

        if models_set_to_train:
            active_groups = list(set([self.ds.get_attr_group(attr) for attr in self.ds.get_active_attributes()]
                                     ).difference(set(models_set_to_train))
                                 )
        else:
            active_groups = [self.ds.get_attr_group(attr) for attr in self.ds.get_active_attributes()]

        attrs_from_active_groups = [attr for group_attr, attr_list in groups.items()
                                    if group_attr in active_groups for attr in attr_list]

        for attr1 in attrs_from_active_groups:
            for attr2 in curr_pair_stats[attr1]:
                # Get distributions from the pair stats.
                past_pdf_attr1, curr_pdf_attr1 = self._attr_pair_stats_to_pdf(self.pair_stats[attr1][attr2],
                                                                              curr_pair_stats[attr1][attr2])

                if past_pdf_attr1 is None or curr_pdf_attr1 is None:
                    continue

                # Computes the K-L divergence.
                kl_attr1 = entropy(past_pdf_attr1, curr_pdf_attr1)

                # We retrain if the KL indicates that any attribute in the group changed more than the threshold.
                if kl_attr1 > thresh:
                    group_attr = self.ds.get_attr_group(attr1)
                    models_to_train.append(group_attr)
                    for att in groups[group_attr]:
                        # Sets the stats for attr1 to be the stats of the dataset used for retraining.
                        self.pair_stats[att] = curr_pair_stats[att]

                        if verbose:
                            logging.debug('model=%s[%s], kl(%s, %s)=%.8f', group_attr, att, attr1, attr2,
                                          kl_attr1)
                    break

        return models_to_train

    def get_models_to_train_weighted_kl(self, thresh, models_set_to_train=None, verbose=False):
        """
        Gets the lists of attributes that will be considered for training.
        This strategy selects a model to be trained if the summation of the KL weighted by the correlation for the
        attributes in the grouped model is higher than the threshold.
        :param verbose:
        :param models_set_to_train:
        :param thresh:
        :return:
        """
        groups = self.ds.get_model_groups()
        _, _, curr_pair_stats = self.ds.get_statistics()

        if self.pair_stats is None:
            logging.debug('No loaded stats to compare. Train all models.')
            # Sets self.pair_stats to the current stats to be eventually saved later.
            self.pair_stats = curr_pair_stats
            return [att for att in groups.keys()]

        if models_set_to_train is not None:
            models_to_train = models_set_to_train
        else:
            models_to_train = []

        if models_set_to_train:
            active_groups = list(set([self.ds.get_attr_group(attr) for attr in self.ds.get_active_attributes()]
                                     ).difference(set(models_set_to_train))
                                 )
        else:
            active_groups = [self.ds.get_attr_group(attr) for attr in self.ds.get_active_attributes()]

        attrs_from_active_groups = [attr for group_attr, attr_list in groups.items()
                                    if group_attr in active_groups for attr in attr_list]

        corr_tuple = self.ds.get_correlations()
        corr = {}
        for attr_tuple in corr_tuple:
            attr1 = attr_tuple[0]
            if attr1 in attrs_from_active_groups:
                corr[attr1] = {}
                for corr_attr_tuple in attr_tuple[1]:
                    attr2 = corr_attr_tuple[0]
                    corr[attr1][attr2] = corr_attr_tuple[1]

        weighted_kl = {}
        for attr1 in attrs_from_active_groups:
            kl_attr1 = {}
            for attr2 in curr_pair_stats[attr1]:
                # Get distributions from the pair stats.
                past_pdf_attr1, curr_pdf_attr1 = self._attr_pair_stats_to_pdf(self.pair_stats[attr1][attr2],
                                                                              curr_pair_stats[attr1][attr2])

                if past_pdf_attr1 is None or curr_pdf_attr1 is None:
                    continue

                # Computes the K-L divergence.
                kl_attr1[attr2] = entropy(past_pdf_attr1, curr_pdf_attr1)

            # weighted_kl(attr1) = 1/n_attrs * sum_{attr2 in dataset} kl(attr1, attr2) * corr(attr1, attr2)
            num_attrs = len(kl_attr1)
            suma = sum([kl_attr1[attr2] * corr[attr1][attr2] for attr2 in kl_attr1.keys()])
            weighted_kl[attr1] = suma / num_attrs

        for group_attr in active_groups:
            attrs_in_group = groups[group_attr]
            # We retrain if the "aggregated" KL weighted by the attribute correlations is larger than the threshold
            # multiplied by the number of attributes (to keep the threshold value "proportional" to a single attribute).
            if sum([weighted_kl[att] for att in attrs_in_group]) > thresh * len(attrs_in_group):
                models_to_train.append(group_attr)
                for att in attrs_in_group:
                    # Sets stats1 for attr1 to be the stats of the dataset used for retraining.
                    self.pair_stats[att] = curr_pair_stats[att]

                    if verbose:
                        logging.debug('model=%s[%s], weighted_kl=%.8f', group_attr, att, weighted_kl[att])

        return models_to_train

    def load_last_training_stats(self, base_path='/tmp/'):
        with open(base_path + self.ds.raw_data.name + '_last_training_stats.ujson',
                  encoding='utf-8') as f:
            self.pair_stats = ujson.load(f)

    def save_last_training_stats(self, base_path='/tmp/'):
        with open(base_path + self.ds.raw_data.name + '_last_training_stats.ujson', 'w',
                  encoding='utf-8') as f:
            ujson.dump(self.pair_stats, f, ensure_ascii=False)
