import logging
import time

import pandas as pd
import numpy as np

from .featurize import FeaturizedDataset
from .learn import RepairModel
from dataset import AuxTables


class RepairEngine:
    def __init__(self, env, dataset):
        self.ds = dataset
        self.env = env
        self.repair_model = {}

    def setup_featurized_ds(self, featurizers):
        tic = time.clock()
        self.feat_dataset = FeaturizedDataset(self.ds, self.env, featurizers)
        toc = time.clock()
        status = "DONE setting up featurized dataset."
        feat_time = toc - tic
        return status, feat_time

    def setup_repair_model(self):
        tic = time.clock()
        feat_info = self.feat_dataset.featurizer_info
        for attr in self.ds.get_active_attributes():
            output_dim = self.feat_dataset.classes[attr]
            self.repair_model[attr] = RepairModel(self.env, feat_info, output_dim, self.ds.is_first_batch(),
                                                  bias=self.env['bias'],
                                                  layer_sizes=self.env['layer_sizes'])
            if self.env['save_load_checkpoint'] and not self.ds.is_first_batch():
                try:
                    self.repair_model[attr].load_checkpoint('/tmp/checkpoint-' + self.ds.raw_data.name + '-' + attr)
                except OSError:
                    logging.info('No existing checkpoint could be loaded for %s.', attr)

        toc = time.clock()
        status = "DONE setting up repair model."
        setup_time = toc - tic
        return status, setup_time

    def fit_repair_model(self):
        tic = time.clock()
        total_training_cells = 0
        X_train, Y_train, mask_train, train_cid = self.feat_dataset.get_training_data()
        for attr in self.ds.get_active_attributes():
            logging.info('Training model for %s with %d training examples (cells)', attr, X_train[attr].size(0))
            tic_attr = time.clock()

            if self.env['save_load_checkpoint'] and not self.ds.is_first_batch():
                tic_skip = time.clock()
                logging.debug("Checking if learning can be skipped for %s...", attr)
                grdt = Y_train[attr].numpy().flatten()
                Y_pred = self.repair_model[attr].infer_values(X_train[attr], mask_train[attr])
                Y_assign = Y_pred.data.numpy().argmax(axis=1)
                accuracy = 100. * np.mean(Y_assign == grdt)
                logging.debug("Previous model accuracy: %.2f. Inference time: %.2f", accuracy, time.clock() - tic_skip)
                if accuracy >= self.env['skip_training_thresh']:
                    logging.debug("Training skipped.")
                    continue

            self.repair_model[attr].fit_model(X_train[attr], Y_train[attr], mask_train[attr], self.env['epochs'])

            if self.env['save_load_checkpoint']:
                tic_checkpoint = time.clock()
                self.repair_model[attr].save_checkpoint('/tmp/checkpoint-' + self.ds.raw_data.name + '-' + attr)
                logging.debug("Checkpointing time %.2f.", time.clock() - tic_checkpoint)

            logging.info('Done. Elapsed time: %.2f', time.clock() - tic_attr)
            total_training_cells += X_train[attr].size(0)
        toc = time.clock()

        if self.env['ignore_previous_training_cells']:
            self._save_training_cells(train_cid)

        status = "DONE training repair model."
        train_time = toc - tic
        return status, train_time, total_training_cells

    def fit_validate_repair_model(self, eval_engine, validate_period):
        tic = time.clock()
        X_train, Y_train, mask_train, train_cid = self.feat_dataset.get_training_data()
        total_training_cells = sum(t.size(0) for t in X_train.values())

        # Training loop
        for epoch_idx in range(1, self.env['epochs']+1):
            logging.info("Repair and validate epoch %d of %d", epoch_idx, self.env['epochs'])
            for attr in self.ds.get_active_attributes():
                logging.info('Training model for %s with %d training examples (cells)', attr, X_train[attr].size(0))
                tic_attr = time.clock()
                self.repair_model[attr].fit_model(X_train[attr], Y_train[attr], mask_train[attr], 1)
                logging.info('Done. Elapsed time: %.2f', time.clock() - tic_attr)
    
            if epoch_idx % validate_period == 0:
                logging.info("Running validation")
                self.infer_repairs()
                report, _, _ = eval_engine.eval_report()
                logging.info(report)
                logging.info("Feature weights:")
                weights, _ = self.get_featurizer_weights()
                logging.info(weights)

        toc = time.clock()

        if self.env['ignore_previous_training_cells']:
            self._save_training_cells(train_cid)

        status = "DONE training repair model."
        train_time = toc - tic
        return status, train_time, total_training_cells

    def _save_training_cells(self, train_cid):
        # Generates table 'training_cells'.
        training_cells = []
        for i in range(len(train_cid)):
            training_cells.append({'_cid_': train_cid[i]})

        self.ds.generate_aux_table(AuxTables.training_cells,
                                   pd.DataFrame(data=training_cells),
                                   store=True,
                                   index_attrs=False,
                                   append=True)


    def infer_repairs(self):
        tic = time.clock()
        X_pred, mask_pred, infer_idx = self.feat_dataset.get_infer_data()
        Y_pred = {}
        for attr in self.ds.get_active_attributes():
            logging.debug('Inferring %d instances of attribute %s', X_pred[attr].size(0), attr)
            tic_attr = time.clock()
            Y_pred[attr] = self.repair_model[attr].infer_values(X_pred[attr], mask_pred[attr])
            logging.debug('Done. Elapsed time: %.2f', time.clock() - tic_attr)

        distr_df, infer_val_df = self.get_infer_dataframes(infer_idx, Y_pred)
        self.ds.generate_aux_table(AuxTables.cell_distr, distr_df, store=True, index_attrs=['_vid_'])
        self.ds.generate_aux_table(AuxTables.inf_values_idx, infer_val_df, store=True, index_attrs=['_vid_'])
        toc = time.clock()
        status = "DONE inferring repairs."
        infer_time = toc - tic
        return status, infer_time

    def get_infer_dataframes(self, infer_idx, Y_pred):
        distr = []
        infer_val = []

        # Need to map the inferred value index of the random variable to the actual value
        # val_idx = val_id - 1 since val_id was numbered starting from 1 whereas
        # val_idx starts at 0.
        query = "SELECT _vid_, val_id-1, rv_val, attribute FROM {pos_values} " \
                "ORDER BY _vid_".format(pos_values=AuxTables.pos_values.name)
        pos_values = self.ds.engine.execute_query(query)

        # Attribute-specific dict mapping from the attribute-specific vid ('attr_vid') to the actual vid
        # and from the value ids ('val_id') for the vid the to the corresponding values ('val').
        # Counters match attribute-specific vids because the we query the pos_values table ordered by vid.
        attr_vid_to_val = {attr: {} for attr in self.ds.get_active_attributes()}

        # Attribute-specific vid counter
        vid_count = {attr: -1 for attr in self.ds.get_active_attributes()}

        prev_vid = None
        for vid, val_idx, val, attr in pos_values:
            if vid != prev_vid:
                vid_count[attr] += 1
                prev_vid = vid
                attr_vid_to_val[attr][vid_count[attr]] = attr_vid_to_val[attr].get(vid_count[attr],
                                                                                   {'vid': vid, 'val_id': {}})
            attr_vid_to_val[attr][vid_count[attr]]['val_id'][val_idx] = val

        for attr in self.ds.get_active_attributes():
            Y_assign = Y_pred[attr].data.numpy().argmax(axis=1)
            for idx in range(Y_pred[attr].shape[0]):
                attr_vid = int(infer_idx[attr][idx])
                rv_distr = list(Y_pred[attr][idx].data.numpy())
                rv_val_idx = int(Y_assign[idx])
                rv_val = attr_vid_to_val[attr][attr_vid]['val_id'][rv_val_idx]
                rv_prob = Y_pred[attr][idx].data.numpy().max()
                vid = attr_vid_to_val[attr][attr_vid]['vid']
                distr.append({'_vid_': vid, 'distribution': [str(p) for p in rv_distr]})
                infer_val.append({'_vid_': vid, 'inferred_val_idx': rv_val_idx, 'inferred_val': rv_val, 'prob':rv_prob})
        distr_df = pd.DataFrame(data=distr)
        infer_val_df = pd.DataFrame(data=infer_val)
        return distr_df, infer_val_df

    def get_featurizer_weights(self):
        tic = time.clock()
        report = {}
        df_per_attr = []
        for attr in self.ds.get_active_attributes():
            report[attr], df_list = self.repair_model[attr]\
                .get_featurizer_weights(self.feat_dataset.featurizer_info)
            # One df per featurizer in df_list.
            df = pd.concat(df_list)
            df.insert(loc=1, column='attribute', value=[attr] * len(df.index))
            df.insert(loc=0, column='batch', value=[self.env['current_batch_number'] + 1] * len(df.index))
            df_per_attr.append(df)
        complete_df = pd.concat(df_per_attr)
        toc = time.clock()
        report_time = toc - tic
        return report, report_time, complete_df
