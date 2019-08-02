import logging
import math

import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import Parameter, ParameterList
from torch.nn.functional import softmax
from tqdm import tqdm
import numpy as np


class TiedLinear(torch.nn.Module):
    """
    TiedLinear is a linear layer with shared parameters for features between (output) classes
    that takes as input a tensor with dimensions (cells) x (output_dim) x (in_features),
    where:
        'cells' is the number of cells selected either for training or to be inferred,
        'output_dim' is the desired output dimension, i.e. the number of classes (maximum domain size), and
        'in_features' is the number of features (from all featurizers used) with shared weights across the classes.
    """

    def __init__(self, env, feat_info, output_dim, bias=False):
        super(TiedLinear, self).__init__()
        self.env = env

        # Initial parameters.
        self.in_features = 0.0
        self.weight_list = ParameterList()
        if bias:
            self.bias_list = ParameterList()
        else:
            self.register_parameter('bias', None)
        self.output_dim = output_dim
        self.bias_flag = bias

        # Iterate over featurizer info list.
        for feat_entry in feat_info:
            learnable = feat_entry.learnable
            feat_size = feat_entry.size
            init_weight = feat_entry.init_weight
            self.in_features += feat_size

            feat_weight = Parameter(init_weight * torch.ones(1, feat_size), requires_grad=learnable)
            if learnable:
                self.reset_parameters(feat_weight)
            self.weight_list.append(feat_weight)

            if bias:
                feat_bias = Parameter(torch.zeros(1, feat_size), requires_grad=learnable)
                if learnable:
                    self.reset_parameters(feat_bias)
                self.bias_list.append(feat_bias)

        self.w = None
        self.b = None

    @staticmethod
    def reset_parameters(tensor):
        stdv = 1.0 / math.sqrt(tensor.size(0))
        tensor.data.uniform_(-stdv, stdv)

    # noinspection PyTypeChecker
    def concat_weights(self):
        self.w = torch.cat([t for t in self.weight_list], -1)

        # Normalize weights.
        if self.env['weight_norm']:
            self.w = self.w.div(self.w.norm(p=2))

        # Expand so we can do matrix multiplication with each cell and their maximum domain size.
        self.w = self.w.expand(self.output_dim, -1)

        if self.bias_flag:
            self.b = torch.cat([t.expand(self.output_dim, -1) for t in self.bias_list], -1)

    def forward(self, x, index, mask):
        # Concatenates different featurizer weights.
        # Needs to be called every pass because the weights might have been updated in previous epochs.
        self.concat_weights()

        # Although 'x' is 3D and 'w' is 2D, the tensors are broadcasted first so that both are 3D.
        # Then, each element in 'x' is multiplied by the corresponding element in 'w'.
        output = x.mul(self.w)

        if self.bias_flag:
            output += self.b

        # The cells are summed along the features' dimension, which yields a 2D output (cells, classes).
        output = output.sum(2)

        # Add our mask so that invalid domain classes for a given variable/_vid_ have a large negative value,
        # resulting in a softmax probability of 0 for such invalid cells.
        output.index_add_(0, index, mask)
        return output


class RepairModel:
    def __init__(self, env, feat_info, output_dim, bias=False):
        self.env = env
        # A list of tuples (name, number_of_features, is_learnable, init_weight, feature_names (list)),
        # one for each featurizer.
        self.feat_info = feat_info
        # Number of classes.
        self.output_dim = output_dim
        self.model = TiedLinear(self.env, feat_info, output_dim, bias)
        self.featurizer_weights = {}

    def fit_model(self, x_train, y_train, mask_train):
        n_examples, n_classes, n_features = x_train.shape

        loss = torch.nn.CrossEntropyLoss()

        trainable_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.env['optimizer'] == 'sgd':
            optimizer = optim.SGD(trainable_parameters,
                                  lr=self.env['learning_rate'],
                                  momentum=self.env['momentum'],
                                  weight_decay=self.env['weight_decay'])
        else:
            optimizer = optim.Adam(trainable_parameters,
                                   lr=self.env['learning_rate'],
                                   weight_decay=self.env['weight_decay'])

        batch_size = self.env['batch_size']
        epochs = self.env['epochs']

        for i in tqdm(range(epochs)):
            cost = 0.
            num_batches = n_examples // batch_size

            for k in range(num_batches):
                start, end = k * batch_size, (k + 1) * batch_size
                cost += self.__train__(loss, optimizer, x_train[start:end], y_train[start:end], mask_train[start:end])

            if self.env['verbose']:
                # Compute and print accuracy at the end of epoch.
                grdt = y_train.numpy().flatten()
                y_pred = self.__predict__(x_train, mask_train)
                y_assign = y_pred.data.numpy().argmax(axis=1)
                logging.debug("Epoch %d: Cost = %f, Accuracy = %.2f%%",
                              i + 1,
                              cost / num_batches,
                              100. * np.mean(y_assign == grdt))

    def infer_values(self, x_pred, mask_pred):
        logging.info('Inferring %d examples (cells)', x_pred.shape[0])
        output = self.__predict__(x_pred, mask_pred)
        return output

    # noinspection PyUnresolvedReferences,PyTypeChecker
    def __train__(self, loss, optimizer, x_train, y_train, mask_train):
        x_var = Variable(x_train, requires_grad=False)
        y_var = Variable(y_train, requires_grad=False)
        mask_var = Variable(mask_train, requires_grad=False)

        index = torch.LongTensor(range(x_var.size()[0]))
        index_var = Variable(index, requires_grad=False)

        optimizer.zero_grad()
        # Fully-connected layer with shared parameters between output classes
        # for linear combination of input features.
        # Mask makes invalid output classes have a large negative value so
        # to zero out softmax probability.
        fx = self.model.forward(x_var, index_var, mask_var)
        # loss is CrossEntropyLoss: combines log softmax + Negative log likelihood loss.
        # Y_Var is just a single 1D tensor with value (0 - 'class' - 1) i.e.
        # index of the correct class ('class' = max domain)
        # fx is a tensor of length 'class' the linear activation going in the softmax.
        output = loss.forward(fx, y_var.squeeze(1))
        output.backward()
        optimizer.step()
        cost = output.item()
        return cost

    # noinspection PyUnresolvedReferences,PyTypeChecker
    def __predict__(self, x_pred, mask_pred):
        x_var = Variable(x_pred, requires_grad=False)
        index = torch.LongTensor(range(x_var.size()[0]))
        index_var = Variable(index, requires_grad=False)
        mask_var = Variable(mask_pred, requires_grad=False)
        fx = self.model.forward(x_var, index_var, mask_var)
        output = softmax(fx, 1)
        return output

    def get_featurizer_weights(self, feat_info):
        report = ""
        for i, f in enumerate(feat_info):
            this_weight = self.model.weight_list[i].data.numpy()[0]
            weight_str = "\n".join("{name} {weight}".format(name=name, weight=weight)
                                   for name, weight in
                                   zip(f.feature_names, map(str, np.around(this_weight, 3))))
            feat_name = f.name
            feat_size = f.size
            max_w = max(this_weight)
            min_w = min(this_weight)
            mean_w = float(np.mean(this_weight))
            abs_mean_w = float(np.mean(np.absolute(this_weight)))
            # Create report
            report += "featurizer %s,size %d,max %.4f,min %.4f,avg %.4f,abs_avg %.4f,weights:\n%s\n" % (
                feat_name, feat_size, max_w, min_w, mean_w, abs_mean_w, weight_str
            )
            # Wrap in a dictionary.
            self.featurizer_weights[feat_name] = {
                'max': max_w,
                'min': min_w,
                'avg': mean_w,
                'abs_avg': abs_mean_w,
                'weights': this_weight,
                'size': feat_size
            }
        return report

    def save_model(self):
        torch.save(self.model.state_dict(), '/tmp/model')

    def load_model(self):
        self.model.load_state_dict(torch.load('/tmp/model'))
        # self.model.eval()
