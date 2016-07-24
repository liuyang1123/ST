from dateutil.parser import parse

ANY = ""
USER = "user"
START_TIME = "start_time"
END_TIME = "end_time"
DURATION = "duration"
DISTANCE = "distance"
MODE_OF_COMMUNICATION = "ModeOfCommunicationPreference"


class Preference(object):  # Python 2
    """
    Clase padre
    """
    preference_name = ""
    event_type = ""

    def __init__(self, preference_name, event_type=ANY, args=dict()):
        self.preference_name = preference_name
        self.event_type = event_type
        self.user = args.get(USER, None)
        self.preference_type = ""

    def _calculate_confidence_score(self, event):
        return

    def confidence_score(self, event):
        """
        Calculates the cosine / distance between the event value and the
        preferred value.
        """
        if self.event_type == ANY or self.event_type == str(event.event_type):
            return self._calculate_confidence_score(event)
        return 0


class BookableHoursPreference(Preference):

    def __init__(self, preference_name, event_type=ANY, args=dict()):
        super(
            BookableHoursPreference,
            self).__init__(
            preference_name,
            event_type,
            args)
        # super().__init__(preference_name, event_type, args)
        self.start = parse(args.get(START_TIME)).time()
        self.end = parse(args.get(END_TIME)).time()
        self.preference_type = "BookableHoursPreference"

    def _calculate_confidence_score(self, event):
        """
        Calculates the cosine / distance between the event value and the
        preferred value.
        """
        if not event.is_in_between(self.start, self.end):
            return event.time_distance(self.start, self.end)
        return 0.0


class DoNotDisturbPreference(Preference):

    def __init__(self, preference_name, event_type=ANY, args=dict()):
        super(
            DoNotDisturbPreference,
            self).__init__(
            preference_name,
            event_type,
            args)
        # super().__init__(preference_name, event_type, args)
        self.start = args.get(START_TIME, None)
        self.end = args.get(END_TIME, None)
        self.preference_type = "DoNotDisturbPreference"

    def _calculate_confidence_score(self, event):
        """
        Calculates the cosine / distance between the event value and the
        preferred value.
        """
        if event.is_in_between(self.start, self.end):
            return event.time_distance(self.start, self.end)
        return 0.0


class DurationPreference(Preference):

    def __init__(self, preference_name, event_type=ANY, args=dict()):
        super(
            DurationPreference,
            self).__init__(
            preference_name,
            event_type,
            args)
        # super().__init__(preference_name, event_type, args)
        self.duration = args.get(DURATION, None)
        self.preference_type = "DurationPreference"

    def _calculate_confidence_score(self, event):
        """
        Calculates the cosine / distance between the event value and the
        preferred value.
        """
        m = max(event.duration, self.duration)
        return -abs(event.duration / m - self.duration / m)


class TimeBetweenPreference(Preference):
    # TODO Add to the calendar api an endpoint to know the meeting before and
    # after a certain time

    def __init__(self, preference_name, event_type=ANY, args=dict()):
        super(
            TimeBetweenPreference,
            self).__init__(
            preference_name,
            event_type,
            args)
        # super().__init__(preference_name, event_type, args)
        self.duration = args.get(DURATION, None)
        self.preference_type = "TimeBetweenPreference"

    def _calculate_confidence_score(self, event):
        """
        Calculates the cosine / distance between the event value and the
        preferred value.
        """
        # Buscar el evento anterior (x) y posterior (y) en el calendario.
        # _x = event.start - x.end
        # _y = y.start - event.end
        # dist = 0.
        # if _x <= duration:
        #   dist += la distancia normalizada
        # if _y <= duration:
        #   dist += la distancia normalizada
        # retornar -abs(dist)
        return 0


class MaxDistancePreference(Preference):

    def __init__(self, preference_name, event_type=ANY, args=dict()):
        # super().__init__(preference_name, event_type, args)
        super(
            MaxDistancePreference,
            self).__init__(
            preference_name,
            event_type,
            args)
        self.distance = args.get(DISTANCE, None)
        self.preference_type = "MaxDistancePreference"

    def _calculate_confidence_score(self, event):
        """
        Calculates the cosine / distance between the event value and the
        preferred value.
        """
        score = 0.0
        dist = self.user.get_location().distance(event.location)
        if dist > self.distance:
            score += -abs(1 - self.distance / dist)
        else:
            score += -abs(1 - dist / self.distance)
        return score


class ModeOfCommunicationPreference(Preference):

    def __init__(self, preference_name, event_type=ANY, args=dict()):
        super(
            ModeOfCommunicationPreference,
            self).__init__(
            preference_name,
            event_type,
            args)
        # super().__init__(preference_name, event_type, args)
        # This should be a vector with probabilities.
        self.mode = args.get(MODE_OF_COMMUNICATION, None)
        self.preference_type = "ModeOfCommunicationPreference"

    def _calculate_confidence_score(self, event):
        """
        Calculates the cosine / distance between the event value and the
        preferred value.
        """
        return 0.
