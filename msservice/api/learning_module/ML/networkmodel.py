class NetworkModel(object):
    network_type = ""

    def __init__(self, instance_id, config=None):
        self.instance_id = instance_id
        self.config = config
        self.model = None

    def train(self, dataset):
        """
        Start the learning algorithm
        """
        return

    def save(self):
        """
        Allows to save the training results, in order to restore them for later use
        """
        return

    def predict(self, query):
        return

    def _load(self):
        return

    def _process_dataset(self, dataset):
        return

    def _build(self):
        """
        Constructs the model
        """
        return

    def _default(self):
        return
