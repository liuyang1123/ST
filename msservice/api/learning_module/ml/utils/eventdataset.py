from .dataset import Dataset


class EventDataset(Dataset):

    def _process(self):
        return


from dateutil.parser import parse
from api.event_module.calendar_client import CalendarDBClient
from api.models import Training
from api.config import SLOT_SIZE, SLOTS_PER_HOUR, SLOTS_PER_DAY
from api.event_module.manager import TRAIN_EVENT_TYPES_DICT


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


def parse_dataset_cf(events, training_objs=[]):
    """
    Values is: [user_id, item_id]
    """
    # Maybe can be added a new model that saves the rating for each timeslot
    # Y se calcula la media de puntuacion de cada timeslot
    # TODO Maybe event_type is a one hot vector?
    values = list()
    labels = list()

    for event in events:
        if is_valid_for_training(event):
            s = parse(event.get('start')).time().hour * SLOTS_PER_HOUR + \
                parse(event.get('start')).time().minute / SLOT_SIZE
            event_type = event.get('categories')
            values.append()  # Get number of event_type id * max_number_of_slots + s -> i * max + offset (AKA item)
            # TODO Here we can append two things: 1. The rating a number
            # between [1-5] or 2. #item / total_items_of_this_category
            labels.append([5])

    for to in training_objs:
        s = to.start.time().hour * SLOTS_PER_HOUR + to.start.time().minute / SLOT_SIZE
        event_type = to.event_type
        values.append()  # Get number of event_type id * max_number_of_slots + s -> i * max + offset (AKA item)
        # TODO Here we can append two things: 1. The rating a number between
        # [1-5] or 2. #item / total_items_of_this_category
        labels.append([1])

    return values, labels


def read_data_sets_cf(
        user=None,
        need_labels=False,
        train_size=80,
        validation_size=10,
        test_size=10):
    """
    1. Get all the user ids
    2. For each user get their events
    3. Call the parser
    4. Return the results
    """
    assert (train_size + validation_size + test_size) == 100
    # Get the historical meetings of the user
    db_client = CalendarDBClient()
    events = db_client.list_all_events(user)
    # Get the training objects saved using the system
    training_objs = Training.objects.filter(user_id=user)
    # Obtain a representation of the data
    data, labels = parse_dataset_cf(events, training_objs)

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
    # , Do the math to return the total number of items
    return data_sets, len(users)
