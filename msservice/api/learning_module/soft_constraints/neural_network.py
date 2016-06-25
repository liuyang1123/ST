import json
from pomegranate import *
from soft_constraints_model import SoftConstraintsModel


class BayesianNetworkModel(SoftConstraintsModel):
    model_type = "BayesianNetwork"

    def _load(self):
        """
        Loads the weights if exists
        """
        pass

    def _build(self, args):
        """

        """
        pass

    def predict(self, args):
        pass

    def score_event(self, event):
        # TODO Armar un diccionario con los datos del evento

        pass

    def train(self, data, labels):
        pass

    def save(self):
        pass

    def _create_default(self):
        """

        """
        return {}

    def _get_distribution_values(self):
        return
