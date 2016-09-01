import requests
from msservice.settings import USER_SERVICE_BASE_URL, US_CLIENT_ID
from api.learning_module.hard_constraints.preferences.manager import UserPreferencesManager
from api.learning_module.soft_constraints.bayesian_network import BayesianNetworkModel
from api.event_module.calendar_client import CalendarDBClient


class Attendee:

    def __init__(self, email=None, user_id=None,
                 sc_model=BayesianNetworkModel):
        self.user_id = None
        self._preferences = []
        self._soft_constraints = None
        if email is not None:
            request = requests.post(USER_SERVICE_BASE_URL +
                                    'api/v1/' +
                                    US_CLIENT_ID +
                                    '/users/get_user_by_email/',
                                    data={"email": email})
            if request.status_code == 200:
                self.user_id = request.json().get('user_id', None)

        if user_id is not None:
            self.user_id = user_id

        if self.user_id is not None:
            self._preferences = self._retrieve_preferences()
            # self._soft_constraints = sc_model(self.user_id)
            # self._soft_constraints.build_model()

    def __str__(self):
        return str(self.user_id)

    def _retrieve_preferences(self):
        result = []
        manager = UserPreferencesManager()
        result = manager.get_list_objects(int(self.user_id))
        manager.close()

        return result

    def get_preference(self, preference_type):
        """
        Given the name of the preference returns it's personal value.
        """
        for pref in self._preferences:
            if pref.preference_type == preference_type:
                return pref

        return None

    def get_score(self, event):
        result = 0.0

        for pref in self._preferences:
            result += pref.confidence_score(event)

        return result / event.number_of_participants()

    def get_prediction(self, event):
        result = 0.0

        if self._soft_constraints is not None:
            # Bayesian Network or NN
            result = self._soft_constraints.score_event(event)

        return result

    def is_available(self, from_date, to_date):
        if self.user_id is None:
            return True

        cl = CalendarDBClient()
        result = cl.available(self.user_id, from_date, to_date)

        return result

    def cleanup(self):
        if self._soft_constraints is not None:
            self._soft_constraints.close()

    def get_location_city(self, datetime_val):
        # TODO Returns the city where the user will be at datetime_val
        return ""

    def get_location(self, datetime_val):
        # TODO Get location if there is information, or predict it
        # returns a Location object
        return ""

    def exists_event(self, event_type, day):
        # T / F
        return
