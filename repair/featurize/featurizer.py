from abc import ABCMeta, abstractmethod
from multiprocessing import Pool


class Featurizer:
    __metaclass__ = ABCMeta

    def __init__(self, learnable=True, init_weight=1.0):
        self.name = None
        self.setup_done = False
        self.learnable = learnable
        self.init_weight = init_weight

        self.ds = None
        self.total_vars = None
        self.classes = None
        self.repair_previous_errors = None
        self._pool = None
        self._batch_size = None

    def setup_featurizer(self, dataset, processes=20, batch_size=32, repair_previous_errors=False):
        self.ds = dataset
        self.total_vars, self.classes = self.ds.get_domain_info()
        self.repair_previous_errors = repair_previous_errors
        # Only create a pool if processes > 1.
        self._pool = Pool(processes) if processes > 1 else None
        self._batch_size = batch_size
        self.setup_done = True
        self.specific_setup()

    @abstractmethod
    def specific_setup(self):
        raise NotImplementedError

    @abstractmethod
    def create_tensor(self):
        """
         This method creates a tensor which has shape
         (# of cells/rvs, maximum size of domain, # of features for this featurizer).

        :return: PyTorch Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def feature_names(self):
        """
        Returns list of human-readable descriptions/names for each feature this featurizer produces.
        """
        raise NotImplementedError

    def _apply_func(self, func, collection):
        if self._pool is None:
            return list(map(func, collection))
        return self._pool.map(func, collection, min(self._batch_size, len(collection)))
