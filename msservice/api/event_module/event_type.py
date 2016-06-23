types_opt = ['breakfast', 'lunch', 'dinner',
             'hangout', 'call', 'meeting']


class EventType:

    def __init__(self, name, is_live, unique_per_day):
        self.name = name
        self.is_live = is_live
        self.unique_per_day = unique_per_day

    def __str__(self):
        return self.name

    def is_live(self):
        return self.is_live

    def is_unique(self):
        return self.unique_per_day
