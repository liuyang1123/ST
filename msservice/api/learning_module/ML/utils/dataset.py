import numpy as np, os, urllib
from abc import ABCMeta, abstractmethod

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
    "movielens": {
        "url": '',
        "name": "movielens",
        "workdir": "./datasets/movielens/",
        "filenames": {

        },
        "one_hot": True,
        "validation_size": 5000
    }
}

class Dataset:
    __metaclass__ = ABCMeta

    def __init__(self, config=None):
        self.url = config.get("url", None)
        self.name = config.get("name", None)
        self.workdir = config.get("workdir", None)
        self.filenames = config.get("filenames", None)
        self.one_hot = config.get("one_hot", False)
        self.validation_size = config.get("validation_size", 1000)

        self._filepaths = self._maybe_download()

        self._process()

        self._index_in_epoch_train = 0
        self._index_in_epoch_test = 0
        self._index_in_epoch_validation = 0

        self._epochs_completed_train = 0
        self._epochs_completed_test = 0
        self._epochs_completed_validation = 0

    def _maybe_download(self):
        """
        Download the data from website, unless it's already here.
        """
        if self.filenames is None:
            return

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
                self.train[0] = self.train[0][perm]
                self.train[1] = self.train[1][perm]
                # Start next epoch
                start = 0
                self._index_in_epoch_train = batch_size
                assert batch_size <= self._num_examples_train
            end = self._index_in_epoch_train
            return self.train[0][start:end], self.train[1][start:end]
        elif dataset == "test":
            start = self._index_in_epoch_test
            self._index_in_epoch_test += batch_size
            if self._index_in_epoch_test > self._num_examples_test:
                # Finished epoch
                self._epochs_completed_test += 1
                # Shuffle the data
                perm = np.arange(self._num_examples_test)
                np.random.shuffle(perm)
                self.test[0] = self.test[0][perm]
                self.test[1] = self.test[1][perm]
                # Start next epoch
                start = 0
                self._index_in_epoch_test = batch_size
                assert batch_size <= self._num_examples_test
            end = self._index_in_epoch_test
            return self.test[0][start:end], self.test[1][start:end]
        elif dataset == "validation":
            start = self._index_in_epoch_validation
            self._index_in_epoch_validation += batch_size
            if self._index_in_epoch_validation > self._num_examples_validation:
                # Finished epoch
                self._epochs_completed_validation += 1
                # Shuffle the data
                perm = np.arange(self._num_examples_validation)
                np.random.shuffle(perm)
                self.validation[0] = self.validation[0][perm]
                self.validation[1] = self.validation[1][perm]
                # Start next epoch
                start = 0
                self._index_in_epoch_validation = batch_size
                assert batch_size <= self._num_examples_validation
            end = self._index_in_epoch_validation
            return self.validation[0][start:end], self.validation[1][start:end]

        raise ValueError('Invalid dataset, options are: train, test, validation')
