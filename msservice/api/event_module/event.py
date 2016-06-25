class Event(object):

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

    def __init__(self, event):
        self.participants = event.participants
        self.event_type = event.event_type
        self.description = event.description
        self.duration = event.duration
        self.start_time = event.start_time
        self.end_time = event.end_time
        self.location = event.location
        self.importance = None  # Calculate lambda
        self.mode_of_communication = None

    def set_mode_of_communication(self, mode_of_communication):
        self.mode_of_communication = mode_of_communication

    def move_time(self, timedelta):
        self.set_start_time(self.start_time + timedelta)
        self.set_end_time(self.end_time + timedelta)

    def get_day(self):
        return self.start_time.date()

    def exists_overlap(self, other_event):
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
        # TODO
        # m1 = max(x1, d1) m2 = max(x2, d2). dist = -(abs(x1-d1)/m1 +
        # abs(x2-d2)/m2)
        return
