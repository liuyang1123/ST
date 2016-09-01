from .networkmodel import NetworkModel


class TensorflowBasedModel(NetworkModel):
    network_type = "TensorflowBasedModel"

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
        return

    def _default(self):
        return
