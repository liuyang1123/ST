class EventType:

    def __init__(self, name, is_live, unique_per_day):
        self._name = name
        self._is_live = is_live
        self._unique_per_day = unique_per_day

    def __str__(self):
        return self._name

    def is_live(self):
        return self._is_live

    def is_unique(self):
        return self._unique_per_day
