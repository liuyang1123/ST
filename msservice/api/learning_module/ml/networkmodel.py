from abc import ABCMeta, abstractmethod

class NetworkModel(object):
    __metaclass__ = ABCMeta
    network_type = ""

    def __init__(self, instance_id, config=None):
        self.instance_id = instance_id
        self.config = config
        self.model = None

    @abstractmethod
    def train(self, dataset):
        """
        Start the learning algorithm
        """
        return

    @abstractmethod
    def save(self):
        """
        Allows to save the training results, in order to restore them for later use
        """
        return

    @abstractmethod
    def predict(self, query):
        return

    @abstractmethod
    def _load(self):
        return

    @abstractmethod
    def _process_dataset(self, dataset):
        return

    @abstractmethod
    def _build(self):
        """
        Constructs the model
        """
        return

    @abstractmethod
    def _default(self):
        return
