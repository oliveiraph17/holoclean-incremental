import logging
import time

import pandas as pd

from .featurize import FeaturizedDataset
from .learn import RepairModel
from dataset import AuxTables


class RepairEngine:
    def __init__(self, env, dataset):
        self.ds = dataset
        self.env = env

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
        output_dim = self.feat_dataset.classes
        self.repair_model = RepairModel(self.env, feat_info, output_dim,
                                        bias=self.env['bias'],
                                        layer_sizes=self.env['layer_sizes'],
                                        self.ds.is_first_batch())
        toc = time.clock()
        status = "DONE setting up repair model."
        setup_time = toc - tic
        return status, setup_time

    def fit_repair_model(self):
        tic = time.clock()
        X_train, Y_train, mask_train, train_cid = self.feat_dataset.get_training_data()
        logging.info('training with %d training examples (cells)', X_train.shape[0])
        self.repair_model.fit_model(X_train, Y_train, mask_train, self.env['epochs'])
        toc = time.clock()

        if self.env['ignore_previous_training_cells']:
            self._save_training_cells(train_cid)

        status = "DONE training repair model."
        train_time = toc - tic
        return status, train_time

    def fit_validate_repair_model(self, eval_engine, validate_period):
        tic = time.clock()
        X_train, Y_train, mask_train, train_cid = self.feat_dataset.get_training_data()
        logging.info('training with %d training examples (cells)', X_train.shape[0])

        # Training loop
        for epoch_idx in range(1, self.env['epochs']+1):
            logging.info("Repair and validate epoch %d of %d", epoch_idx, self.env['epochs'])
            self.repair_model.fit_model(X_train, Y_train, mask_train, 1)

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
        return status, train_time

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
        Y_pred = self.repair_model.infer_values(X_pred, mask_pred)
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
        Y_assign = Y_pred.data.numpy().argmax(axis=1)
        domain_size = self.feat_dataset.var_to_domsize

        # Need to map the inferred value index of the random variable to the actual value
        # val_idx = val_id - 1 since val_id was numbered starting from 1 whereas
        # val_idx starts at 0.
        query = 'SELECT _vid_, val_id-1, rv_val FROM {pos_values}'.format(pos_values=AuxTables.pos_values.name)
        pos_values = self.ds.engine.execute_query(query)
        # dict mapping _vid_ --> val_idx --> value
        vid_to_val = {}
        for vid, val_idx, val in pos_values:
            vid_to_val[vid] = vid_to_val.get(vid, {})
            vid_to_val[vid][val_idx] = val

        for idx in range(Y_pred.shape[0]):
            vid = int(infer_idx[idx])
            rv_distr = list(Y_pred[idx].data.numpy())
            rv_val_idx = int(Y_assign[idx])
            rv_val = vid_to_val[vid][rv_val_idx]
            rv_prob = Y_pred[idx].data.numpy().max()
            d_size = domain_size[vid]
            distr.append({'_vid_': vid, 'distribution':[str(p) for p in rv_distr[:d_size]]})
            infer_val.append({'_vid_': vid, 'inferred_val_idx': rv_val_idx, 'inferred_val': rv_val, 'prob':rv_prob})
        distr_df = pd.DataFrame(data=distr)
        infer_val_df = pd.DataFrame(data=infer_val)
        return distr_df, infer_val_df

    def get_featurizer_weights(self):
        tic = time.clock()
        report = self.repair_model.get_featurizer_weights(self.feat_dataset.featurizer_info)
        toc = time.clock()
        report_time = toc - tic
        return report, report_time
