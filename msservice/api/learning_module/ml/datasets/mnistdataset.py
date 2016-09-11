import numpy as np
import gzip
from dataset import Dataset


class MNISTDataset(Dataset):
    """
    MNIST dataset
    """

    def _filter_by_range(self, obj, start, end):
        return np.array(obj)[start:end]

    def _shuffle(self, obj, perm):
        return obj[perm]
        
    def _read32(self, bytestream):
        dt = np.dtype(np.uint32).newbyteorder('>')
        return np.frombuffer(bytestream.read(4), dtype=dt)

    def _extract_images(self, filename):
        """
        Extract the images into a 4D uint8 numpy array [index, y, x, depth].
        """
        with gzip.open(filename) as bytestream:
            magic = self._read32(bytestream)
            if magic != 2051:
                raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, filename))
            num_images = self._read32(bytestream)  # N images
            rows = self._read32(bytestream)  # 28
            cols = self._read32(bytestream)  # 28
            buf = bytestream.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(num_images, rows, cols, 1)
            return data

    def _dense_to_one_hot(self, labels_dense, num_classes=10):
        """
        Convert class labels from scalars to one-hot vectors.
        """
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    def _extract_labels(self, filename, one_hot=False):
        """
        Extract the labels into a 1D uint8 numpy array [index].
        """
        with gzip.open(filename) as bytestream:
            magic = self._read32(bytestream)
            if magic != 2049:
                raise ValueError(
                    'Invalid magic number %d in MNIST label file: %s' %
                    (magic, filename))
            num_items = self._read32(bytestream)
            buf = bytestream.read(num_items)
            labels = np.frombuffer(buf, dtype=np.uint8)
            if one_hot:
                return self._dense_to_one_hot(labels)
            return labels

    def _process(self):
        train_images = self._extract_images(
            self.workdir + self.filenames["TRAIN_IMAGES"])
        train_labels = self._extract_labels(
            self.workdir + self.filenames["TRAIN_LABELS"],
            one_hot=self.one_hot)
        self.train = [train_images[self.validation_size:],
                      train_labels[self.validation_size:]]

        test_images = self._extract_images(
            self.workdir + self.filenames["TEST_IMAGES"])
        test_labels = self._extract_labels(
            self.workdir + self.filenames["TEST_LABELS"],
            one_hot=self.one_hot)
        self.test = [test_images, test_labels]

        self.validation = [train_images[:self.validation_size],
                           train_labels[:self.validation_size]]

        # assert self.train[0].shape[0] == self.train[1].shape[0], (
        #     "images.shape: %s labels.shape: %s" % (self.train[0].shape,
        #                                            self.train[1].shape))

        self._num_examples_train = self.train[0].shape[0]
        self._num_examples_test = self.test[0].shape[0]
        self._num_examples_validation = self.validation[0].shape[0]
        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        # assert images.shape[3] == 1
        # Convert from [0, 255] -> [0.0, 1.0].
        self.train[0] = self.train[0].reshape(
            self.train[0].shape[0],
            self.train[0].shape[1] * self.train[0].shape[2])
        self.train[0] = self.train[0].astype(np.float32)
        self.train[0] = np.multiply(self.train[0], 1.0 / 255.0)

        self.test[0] = self.test[0].reshape(
            self.test[0].shape[0],
            self.test[0].shape[1] * self.test[0].shape[2])
        self.test[0] = self.test[0].astype(np.float32)
        self.test[0] = np.multiply(self.test[0], 1.0 / 255.0)

        self.validation[0] = self.validation[0].reshape(
            self.validation[0].shape[0],
            self.validation[0].shape[1] * self.validation[0].shape[2])
        self.validation[0] = self.validation[0].astype(np.float32)
        self.validation[0] = np.multiply(self.validation[0], 1.0 / 255.0)
