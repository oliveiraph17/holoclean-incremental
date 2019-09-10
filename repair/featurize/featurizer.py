from abc import ABCMeta, abstractmethod
from multiprocessing import Pool


class Featurizer:
    __metaclass__ = ABCMeta

    def __init__(self, learnable=True, init_weight=1.0, **kwargs):
        self.name = None
        self.setup_done = False
        self.learnable = learnable
        self.init_weight = init_weight
        self.addn_kwargs = kwargs

    def setup_featurizer(self, env, dataset):
        self.env = env
        self.ds = dataset
        self.total_vars, self.classes = self.ds.get_domain_info()
        # only create a pool if processes > 1
        self._pool = Pool(env['threads']) if env['threads'] > 1 else None
        self._batch_size = env['batch_size']
        self.setup_done = True
        self.specific_setup()

    @abstractmethod
    def specific_setup(self):
        raise NotImplementedError

    @abstractmethod
    def gen_feat_tensor(self, vid):
        """
        Generates a torch.Tensor(max_domain, num_features) by featurizing the cell
        with vid == :param vid:.
        """
        raise NotImplementedError

    @abstractmethod
    def feature_names(self):
        """
        Returns list of human-readable description/names for each feature
        this featurizer produces.
        """
        raise NotImplementedError

    def num_features(self):
        return len(self.feature_names())
