# TODO Datafeeder para tf y bn, normalizacion, Panda?
# Autoencoder
import numpy as np
from api.event_module.calendar_client import CalendarDBClient
from api.models import Training

ModeOfCommunication = "ModeOfCommunication"


class DataSet:

    def __init__(self, data, labels, with_labels=False):
        self._input, self._labels = self._process(data, labels)
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._with_labels = with_labels
        self._number_of_inputs = len(self._input)

    def _process(self, data, labels):
        """
        For normalization purposes.
        """
        return data, labels

    def num_examples(self):
        return self._number_of_inputs

    def input_data(self):
        return self._input

    def labels(self):
        return self._labels

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
        if self._with_labels:
            return self._input[start:end], self._labels[start:end]
        return self._input[start:end], []


class DataSets:
    pass


def is_valid_for_training(event):
    # TODO An event is valid when all the information is complete.
    return False


def parse(events, training_objs=[], with_labels=False):
    """
    Returns a list of vectors:
        [ [event_type, duration, day, timeslot, location, accepted] ]
          ['call', 30, 'Monday', '9:00', 1, True]
    We don't take into consideration location yet.
    """
    values = list()
    labels = list()

    for event in events:
        if is_valid_for_training(event):
            # values.append([
            #     event_type,
            #     duration,
            #     day,
            #     timeslot,
            #     location,
            #     accepted
            # ])
            # timeslot_number = e['start'].time().hour * 12 + \
            #     e['start'].time().minute / 5
            # # To recover the time -> hour = X/12, resto_hour = m/5
            # values.append([e['summary'], e['duration'], e['start'].weekday(),
            #                timeslot_number, e['location'], True])
            values.append([])
            if with_labels:
                labels.append([])

    for to in training_objs:
        # ...
        values.append([])
        if with_labels:
            labels.append(to.feedback)

    if with_labels:
        return values, labels
    else:
        return values


def read_data_sets_bn(user):
    db_client = CalendarDBClient()
    events = db_client.list_all_events(user)
    training_objs = Training.objects.filter(attendee=user)
    data = parse(events, training_objs, False)
    # TODO Add the use of Training

    data_sets = DataSets()
    data_sets.train = DataSet(data, [])
    # data_sets.validation = DataSet([], [])
    # data_sets.test = DataSet([], [])
    return data_sets
