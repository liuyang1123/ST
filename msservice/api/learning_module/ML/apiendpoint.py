import json
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework.decorators import detail_route
from rest_framework.renderers import JSONRenderer
from utils.dataset import *
from .bayesiannetwork import BayesianNetworkModel

class MLViewSet(viewsets.ViewSet):
    """
    A viewset that provides the standard actions.
    """
    permission_classes = (AllowAny,)
    renderer_classes = (JSONRenderer, )

    def _get_model(self, model, instance_id, config):
        if model == "BayesianNetwork":
            return BayesianNetworkModel(instance_id, config)
        return None

    def _get_dataset(self, instance_id, config):
        # TODO Change this so the dataset is selected based on a config parameter
        return read_data_sets(instance_id, False)

    @detail_route(methods=['post'])
    def train(self, request, model, pk):
        c = request.data.get("config", None)
        d = self._get_dataset(pk, c)
        m = self._get_model(model, pk, c)

        if m is None:
            return Response({"error": "The model name doesn't exists."},
                            status=status.HTTP_400_BAD_REQUEST)
        m.train(d) # TODO Wrap this in a Celery task

        return Response({"message": "Training started."},
                        status=status.HTTP_200_OK)

    @detail_route(methods=['post'])
    def predict(self, request, model, pk):
        c = request.data.get("config", None)
        m = self._get_model(model, pk, c)    
        obs = request.data.get('query', '{}')
        result = m.predict(obs)

        return Response(json.dumps(result), status=status.HTTP_200_OK)
