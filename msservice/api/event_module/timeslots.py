from datetime import datetime, timedelta
from dateutil.parser import parse
from calendar_client import CalendarDBClient
from api.config import SLOT_SIZE

# TODO Transfer this to calendar_client

class TimeSlot(object): # Python 2
    _start = None
    _end = None

    def __init__(self, start=None, end=None):
        self._start = start
        self._end = end

    def get_start(self):
        return self._start

    def set_start(self, start):
        self._start = start

    def get_end(self):
        return self._end

    def set_end(self, end):
        self._end = end


class TimeSlotManager(object): # Python 2
    """
    """

    def __init__(self, user_id, start_period, end_period, duration):
        self._user_id = user_id
        self._start_period = start_period  # Initial value of the interval
        self._end_period = end_period  # Final value of the interval

        self._iterator = 0
        self.number_of_slots = duration / SLOT_SIZE
        self.slot_delta = timedelta(minutes=SLOT_SIZE)  # 5 Min = size of one slot
        self.duration_delta = self.number_of_slots * self.slot_delta
        self.day_delta = timedelta(days=1)
        self.are_more_days_to_query = False

        self._db_client = CalendarDBClient()
        self._time_slots = self._filter()

    def _filter(self):
        result = []
        busy_space = []

        start = datetime.combine(self._start_period,
                                 datetime.min.time())
        end = datetime.combine(self._start_period, datetime.min.time()) + self.duration_delta
        max_datetime = None

        if self._start_period + self.day_delta <= self._end_period:
            # To make this more efficient, it will only query one day at a time.
            busy_space = self._db_client.free_busy(
                self._user_id, self._start_period,
                self._start_period + self.day_delta)
            max_datetime = datetime.combine(self._start_period + self.day_delta,
                                            datetime.min.time())
            self._start_period = self._start_period + self.day_delta
            self.are_more_days_to_query = True
        else:
            busy_space = self._db_client.free_busy(
                self._user_id, self._start_period, self._end_period)
            self.are_more_days_to_query = False
            max_datetime = datetime.combine(self._end_period,
                                            datetime.min.time())

        self._iterator = 0

        slot_iterator = 0

        while end <= max_datetime:
            if slot_iterator < len(busy_space):
                # Check that the slot is valid
                if end <= parse(busy_space[slot_iterator]["start"]):
                    result.append(TimeSlot(start, end))
                    start = end
                    end = start + self.duration_delta
                else:
                    start = parse(busy_space[slot_iterator]["end"])
                    end = start + self.duration_delta
                    slot_iterator += 1
            else:
                result.append(TimeSlot(start, end))
                start = end
                end = start + self.duration_delta

        return result

    def has_next(self):
        return (self._iterator < len(self._time_slots)) or self.are_more_days_to_query

    def next(self):
        result = None

        if self._iterator < len(self._time_slots):
            result = self._time_slots[self._iterator]
            self._iterator += 1
        elif self.are_more_days_to_query:
            self._time_slots = self._filter()
            result = self.next()

        return result
