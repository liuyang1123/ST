"""
Author:"""
import numpy as np
from dataset import Dataset
from dateutil.parser import parse
from api.event_module.calendar_client import CalendarDBClient
from api.models import RankingEvents, EventItem
from api.config import SLOT_SIZE, SLOTS_PER_HOUR, SLOTS_PER_DAY
from api.event_module.manager import TRAIN_EVENT_TYPES_DICT

class EventsDatasetGenerator(object):
    def __init__(self, need_labels=True, rank=False, initial=False):
        # Bayesian Network config:
        #   need_labels = False, rank = False, initial = True
        # MLP config:
        #   need_labels = True, rank = False, initial = True
        # CF config:
        #   need_labels = True, rank = True, initial = True
        self.db_client = CalendarDBClient()
        self.x = np.array([[]])
        self.y = np.array([[]])
        self.need_labels = need_labels
        self.rank = rank
        self.initial = initial
        self.x = np.array([[]])
        self.y = np.array([[]])

        self._fetch()

    def _to_item_id(self, event):
        ei = EventItem.objects.filter(event_type=event[1],
                                      duration=event[2],
                                      day=event[3],
                                      timeslot=event[4],
                                      location=event[5])
        if ei.exists():
            ei = ei[0]
        else:
            # Save the new element
            ei = EventItem(event_type=event[1],
                           duration=event[2],
                           day=event[3],
                           timeslot=event[4],
                           location=event[5])
            ei.save()

        return ei.id

    def _is_valid_for_training(self, event):
        # An event is valid when all the information is complete.
        if not event.get(
            'deleted', False) and event.get(
            'start', '') != '' and event.get(
                'end', '') != '' and event.get(
                    'duration', -1) != -1 and event.get(
                        'categories', '') != '':
            return True
        return False

    def _parse(self, events, training):
        """
        Returns a list of vectors:
            [[user_id, type, duration, day, timeslot, location, accepted], ...]
            OR
            [[user_id, type, duration, day, timeslot, location, rating], ...]
        We don't take into consideration location yet.
        """
        values = np.array([[]])
        labels = np.array([[]])

        for event in events:
            if self._is_valid_for_training(event):
                s = parse(event.get('start')).time().hour * SLOTS_PER_HOUR + \
                    parse(event.get('start')).time().minute / SLOT_SIZE

                item = [event.get("user_id"),
                        TRAIN_EVENT_TYPES_DICT[event.get('categories')],
                        event.get('duration'),
                        parse(event.get('start')).weekday(),
                        s / SLOTS_PER_DAY,
                        event.get('location', '')]

                if self.rank:
                    # Each item should be like: [User, Item]
                    np.append(values, [[item[0], self._to_item_id(item)]])
                    if self.need_labels:
                        np.append(labels, [[4]])
                else:
                    np.append(values, [item])
                    if self.need_labels:
                        np.append(labels, [[1]]) # [0, 1]

        for to in training:
            s = to.start.time().hour * SLOTS_PER_HOUR + to.start.time().minute / SLOT_SIZE
            item = [to.user_id,
                    TRAIN_EVENT_TYPES_DICT[to.event_type],
                    to.duration,
                    to.start.weekday(),
                    s / SLOTS_PER_DAY,
                    to.location]

            if self.need_labels:
                np.append(labels, [[to.feedback]])

            if self.rank:
                # Each item should be like: [User, Item]
                np.append(values, [[item[0], self._to_item_id(item)]])
            else:
                np.append(values, [item])

            if self.need_labels:
                np.append(labels, [[to.feedback]])

        return values, labels

    def _fetch(self):
        events = []
        if self.initial:
            # Get the historical meetings of the user
            events = self.db_client.list_all_training_events()
        # Get the training objects saved using the system
        training_objs = RankingEvents.objects.all()
        # Obtain a representation of the data
        udata, ulabels = self._parse(events, training_objs)
        self.x = np.append(self.x, udata)
        self.y = np.append(self.y, ulabels)


class EventsDataset(Dataset):
    """
    -
    """

    def _process(self):
        """
        After the files are downloaded, create a representation of the dataset.
        """
        dataset_gen = EventsDatasetGenerator(
            need_labels=self.config.get("need_labels", True),
            rank=self.config.get("rank", False),
            initial=self.config.get("initial", True))
        self.train = [dataset_gen.x, dataset_gen.y]

    def _filter_by_range(self, obj, start, end):
        """
        -
        """
        return np.array(obj)[start:end]

    def _shuffle(self, obj, perm):
        """
        -
        """
        return obj[perm]
