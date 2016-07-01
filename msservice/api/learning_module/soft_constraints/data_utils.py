import numpy as np
from dateutil.parser import parse
from api.event_module.calendar_client import CalendarDBClient
from api.models import Training
from api.config import SLOT_SIZE, SLOTS_PER_HOUR, SLOTS_PER_DAY
from api.event_module.manager import TRAIN_EVENT_TYPES_DICT


class DataSet(object):  # Python
    """

    """

    def __init__(self, values, labels, with_labels=False):
        self._input = values
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._with_labels = with_labels
        self._number_of_inputs = len(self._input)

    @property
    def events(self):
        return self._input

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._number_of_inputs

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size=None):
        """
        Return the next `batch_size` examples from this data set.
        """
        if self._number_of_inputs == 0:
            return [], []

        if not batch_size:
            batch_size = self._number_of_inputs

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._number_of_inputs:
            # Finished epoch
            self._epochs_completed += 1

            # Shuffle the data
            perm = np.arange(self._number_of_inputs)
            np.random.shuffle(perm)
            self._input = self._input[perm]
            if self._with_labels:
                self._labels = self._labels[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._number_of_inputs

        end = self._index_in_epoch
        # If _with_labels is False then it return []
        return self._input[start:end], self._labels[start:end]


class DataSets(object):
    pass


def is_valid_for_training(event):
    # TODO An event is valid when all the information is complete.
    if not event.get(
        'deleted', False) and event.get(
        'start', '') != '' and event.get(
            'end', '') != '' and event.get(
                'duration', -1) != -1 and event.get(
                    'categories', '') != '':
        return True
    return False


def parse_dataset(events, training_objs=[], with_labels=False):
    """
    Returns a list of vectors:
        [ [event_type, duration, day, timeslot, location, accepted] ]
          ['call', 30, 'Monday', '9:00', 1, True]
        labels are a one hot vector [0, 1] -> True [1, 0] -> False
    We don't take into consideration location yet.
    """
    # TODO Maybe event_type is a one hot vector?
    values = list()
    labels = list()

    for event in events:
        if is_valid_for_training(event):
            s = parse(event.get('start')).time().hour * SLOTS_PER_HOUR + \
                parse(event.get('start')).time().minute / SLOT_SIZE
            values.append([
                TRAIN_EVENT_TYPES_DICT[event.get('categories')],
                event.get('duration'),
                parse(event.get('start')).weekday(),
                s / SLOTS_PER_DAY,
                event.get('location', ''),
            ])
            if with_labels:
                labels.append([0, 1])

    for to in training_objs:
        s = to.start.time().hour * SLOTS_PER_HOUR + to.start.time().minute / SLOT_SIZE
        values.append([TRAIN_EVENT_TYPES_DICT[to.event_type], to.duration,
                       to.start.weekday(), s / SLOTS_PER_DAY, to.location])
        if with_labels:
            labels.append([0, 1] if to.feedback else [1, 0])

    return values, labels


def read_data_sets(user, need_labels=False, train_size=100, validation_size=0,
                   test_size=0):
    assert (train_size + validation_size + test_size) == 100
    # Get the historical meetings of the user
    db_client = CalendarDBClient()
    events = db_client.list_all_events(user)
    # Get the training objects saved using the system
    training_objs = Training.objects.filter(user_id=user)
    # Obtain a representation of the data
    data, labels = parse_dataset(events, training_objs, need_labels)

    t_max = len(data) * train_size / 100
    v_max = len(data) * validation_size / 100

    data_sets = DataSets()
    data_sets.train = DataSet(data[:t_max],
                              labels[:t_max] if need_labels else [],
                              need_labels)
    data_sets.validation = DataSet(data[t_max:v_max],
                                   data[t_max:v_max] if need_labels else [],
                                   need_labels)
    data_sets.test = DataSet(data[v_max:],
                             labels[v_max:] if need_labels else [],
                             need_labels)
    return data_sets
