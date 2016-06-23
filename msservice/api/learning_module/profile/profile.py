from api.learning_module import utils


class Attendee(object):

    def __init__(self, preferences):
        self.preferences = preferences

    def get_preference_score(self, event):
        result = 0.0

        for pref in self.preferences:
            result += pref.score(event)

        return result / event.number_of_participants()

    def get_preference(string):
        """
        Given the name of the preference returns it's personal value.
        """
        # TODO - Database
        return

    def get_location(datetime):
        # TODO Get location if there is information, or predict it
        # returns a Location object
        return

    def exists_event(self, event_type, day):
        # T / F
        return
