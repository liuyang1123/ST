import numpy as np
import os
import urllib
from abc import ABCMeta, abstractmethod
from shutil import rmtree

KNOWN_DATASETS = {
    "mnist": {
        "url": 'http://yann.lecun.com/exdb/mnist/',
        "name": "mnist",
        "workdir": "./datasets/mnist/",
        "filenames": {
            "TRAIN_IMAGES": 'train-images-idx3-ubyte.gz',
            "TRAIN_LABELS": 'train-labels-idx1-ubyte.gz',
            "TEST_IMAGES": 't10k-images-idx3-ubyte.gz',
            "TEST_LABELS": 't10k-labels-idx1-ubyte.gz'
        },
        "one_hot": True,
        "validation_size": 5000
    },
    "movielens100k": {
        "url": 'http://files.grouplens.org/datasets/movielens/',
        "name": "movielens100k",
        "workdir": "./datasets/movielens/",
        "filenames": {
            "100k": "ml-100k.zip"
        },
        "one_hot": True,
        "validation_size": 5000
    },
    "events": {
        "url": None,
        "name": "events",
        "workdir": None,
        "filenames": {},
        "one_hot": False,
        "validation_size": 50
    },
    "babi": {
        "url": 'http://www.thespermwhale.com/jaseweston/babi/',
        "name": "babi",
        "workdir": "./datasets/dmn/data/",
        "filenames": {
            "tasks": "tasks_1-20_v1-2.tar.gz"
        },
        "training_task": "1",
        "testing_task": "1",
        "use_10k": False,
        "input_mask_mode": 'sentence',
        "word_vector_length": 50
    },
}


class Dataset:
    __metaclass__ = ABCMeta

    def __init__(self, config={}):
        self.config = config
        self.url = config.get("url", None)
        self.name = config.get("name", None)
        self.workdir = config.get("workdir", None)
        self.filenames = config.get("filenames", None)
        self.one_hot = config.get("one_hot", False)
        self.validation_size = config.get("validation_size", 1000)
        self.restart = config.get("restart", False)

        self._num_examples_train = 0
        self._num_examples_test = 0
        self._num_examples_validation = 0

        self._index_in_epoch_train = 0
        self._index_in_epoch_test = 0
        self._index_in_epoch_validation = 0

        self._epochs_completed_train = 0
        self._epochs_completed_test = 0
        self._epochs_completed_validation = 0

        self.train = [[], []]
        self.test = [[], []]
        self.validation = [[], []]

        self._filepaths = self._maybe_download()

        self._process()


    def _maybe_download(self):
        """
        Download the data from website, unless it's already here.
        """
        if self.filenames is None:
            return

        if self.restart:
            rmtree(self.workdir)

        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)

        filepaths = []

        for filename in self.filenames.values():
            filepath = os.path.join(self.workdir, filename)

            if not os.path.exists(filepath):
                filepath, _ = urllib.urlretrieve(self.url + filename, filepath)
                filepaths.append(filepath)

        return filepaths

    @abstractmethod
    def _process(self):
        """
        After the files are downloaded, create a representation of the dataset.
        """
        return

    @abstractmethod
    def _filter_by_range(self, obj, start, end):
        return

    @abstractmethod
    def _shuffle(self, obj, perm):
        return

    def number_of_examples_train(self):
        return self._num_examples_train

    def number_of_examples_test(self):
        return self._num_examples_test

    def next_batch(self, batch_size, dataset="train"):
        """
        Return the next `batch_size` examples from this data set.
        """
        if dataset == "train":
            start = self._index_in_epoch_train
            self._index_in_epoch_train += batch_size
            if self._index_in_epoch_train > self._num_examples_train:
                # Finished epoch
                self._epochs_completed_train += 1
                # Shuffle the data
                perm = np.arange(self._num_examples_train)
                np.random.shuffle(perm)
                self.train[0] = self._shuffle(self.train[0], perm)
                self.train[1] = self.train[1][perm]
                # Start next epoch
                start = 0
                self._index_in_epoch_train = batch_size
                assert batch_size <= self._num_examples_train
            end = self._index_in_epoch_train
            return self._filter_by_range(self.train[0], start, end), self.train[1][start:end]
        elif dataset == "test":
            start = self._index_in_epoch_test
            self._index_in_epoch_test += batch_size
            if self._index_in_epoch_test > self._num_examples_test:
                # Finished epoch
                self._epochs_completed_test += 1
                # Shuffle the data
                perm = np.arange(self._num_examples_test)
                np.random.shuffle(perm)
                self.test[0] = self._shuffle(self.test[0], perm)
                self.test[1] = self.test[1][perm]
                # Start next epoch
                start = 0
                self._index_in_epoch_test = batch_size
                assert batch_size <= self._num_examples_test
            end = self._index_in_epoch_test
            return self._filter_by_range(self.test[0], start, end), self.test[1][start:end]
        elif dataset == "validation":
            start = self._index_in_epoch_validation
            self._index_in_epoch_validation += batch_size
            if self._index_in_epoch_validation > self._num_examples_validation:
                # Finished epoch
                self._epochs_completed_validation += 1
                # Shuffle the data
                perm = np.arange(self._num_examples_validation)
                np.random.shuffle(perm)
                self.validation[0] = self._shuffle(self.validation[0], perm)
                self.validation[1] = self.validation[1][perm]
                # Start next epoch
                start = 0
                self._index_in_epoch_validation = batch_size
                assert batch_size <= self._num_examples_validation
            end = self._index_in_epoch_validation
            return self._filter_by_range(self.validation[0], start, end), self.validation[1][start:end]

        raise ValueError(
            'Invalid dataset, options are: train, test, validation')
