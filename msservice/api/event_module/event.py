from api.config import SLOT_SIZE

class Event(object): # Python 2

    def __init__(self, participants, event_type=None, description=None,
                 duration=None, start_time=None, end_time=None, location=None):
        self.participants = participants
        self.event_type = event_type
        self.description = description
        self.duration = duration
        self.start_time = start_time
        self.end_time = end_time
        self.location = location
        self.importance = None  # Calculate lambda
        self.mode_of_communication = None

    def set_mode_of_communication(self, mode_of_communication):
        self.mode_of_communication = mode_of_communication

    def move_time(self, timedelta):
        self.set_start_time(self.start_time + timedelta)
        self.set_end_time(self.end_time + timedelta)

    def get_day(self):
        return self.start_time.date()

    def exists_overlap(self):
        """
        Verifica que no exista otro evento, tal que:
            - Alguno de los participantes es tambien participante del otro evento
                Participant(Xi) ^ Participant(Xj) = Vacio
            Time Overlap:
            - El evento se superpone (horario). Todos los participantes estan
                libres a esa hora. Por lo que cumple el punto anterior tambien.
            Room Overlap:
            - El evento es en el mismo lugar. El lugar esta libre a es hora.
                location.is_free()
        Return True or False
        """
        # TODO After improving the Location knowledge
        for p in self.participants:
            if not p.is_available(self.start_time, self.end_time):
                return True
        return False

    def is_a_valid_location(self):
        if self.location is not None and not self.location.supports(
                len(self.participants)):
            return False
        return True

    def number_of_participants(self):
        return len(self.participants)

    def should_be_unique(self):
        return self.event_type.is_unique()

    def set_start_time(self, start_time):
        self.start_time = start_time

    def set_end_time(self, end_time):
        self.end_time = end_time

    def is_live(self):
        return self.event_type.is_live()

    def is_in_between(self, d1, d2):
        return

    def time_distance(self, d1, d2):
        _x1 = self.start_time.time().hour * 12 + self.start_time.time().minute / SLOT_SIZE
        _d1 = d1.hour * 12 + d1.minute / SLOT_SIZE
        _x2 = self.end_time.time().hour * 12 + self.end_time.time().minute / SLOT_SIZE
        _d2 = d2.hour * 12 + d2.minute / SLOT_SIZE
        m1 = max(_x1, _d1)
        m2 = max(_x2, _d2)
        dist = -abs(_x1-_d1)/m1 - abs(_x2-_d2)/m2

        return dist
