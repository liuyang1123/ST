import json
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework.decorators import detail_route
from rest_framework.renderers import JSONRenderer
from datasets.dataset import KNOWN_DATASETS
from datasets.mnistdataset import MNISTDataset
from datasets.movielensdataset import MovieLensDataset
from datasets.babidataset import BABIDataset
from datasets.eventsdataset import EventsDataset
from bayesiannetwork import BayesianNetworkModel
from mlp import MultiLayerPerceptronModel
from cfbaseline import CFBaselineModel
from dmn import DynamicMemoryNetworkModel
from config import MNISTConfig, CFMovieLensConfig

class MLViewSet(viewsets.ViewSet):
    """
    A viewset that provides the standard actions.
    """
    permission_classes = (AllowAny,)
    renderer_classes = (JSONRenderer, )

    def _get_model(self, model, instance_id, config):
        if model == "BayesianNetwork":
            return BayesianNetworkModel(instance_id, config)
        elif model == "MultiLayerPerceptron":
            # TODO Override config based on the config parameter
            c = MNISTConfig()
            return MultiLayerPerceptronModel(instance_id, c)
        elif model == "CFBaseline":
            c = CFMovieLensConfig()
            return CFBaselineModel(instance_id, config=c)
        elif model == "DynamicMemoryNetwork":
            return DynamicMemoryNetworkModel(instance_id, config={})
        return None

    def _get_dataset(self, instance_id, config):
        if config is not None:
            try:
                d = KNOWN_DATASETS[config]
                if d['name'] == 'mnist':
                    return MNISTDataset(config=d)
                elif d['name'] == 'movielens100k':
                    return MovieLensDataset(config=d)
                elif d['name'] == 'events':
                    return EventsDataset(config=d)
                elif d['name'] == 'babi':
                    return BABIDataset(config=d)
            except KeyError:
                pass
        return None

    @detail_route(methods=['post'])
    def train(self, request, model, pk):
        dataset = request.data.get("dataset", None)
        config = request.data.get("config", {})
        d = self._get_dataset(pk, dataset)
        m = self._get_model(model, pk, config)

        if m is None:
            return Response({"error": "The model name doesn't exists."},
                            status=status.HTTP_400_BAD_REQUEST)
        m.train(d)  # TODO Wrap this in a Celery task

        return Response({"message": "Training started."},
                        status=status.HTTP_200_OK)

    @detail_route(methods=['post'])
    def predict(self, request, model, pk):
        c = request.data.get("config", None)
        m = self._get_model(model, pk, c)
        obs = request.data.get('query', {})
        result = m.predict(obs)

        return Response(json.dumps(result), status=status.HTTP_200_OK)
