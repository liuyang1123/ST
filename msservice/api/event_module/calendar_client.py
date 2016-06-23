import requests
from msservice import settings


class CalendarDBClient():

    def __init__(self):
        self.CLIENT_ID = "72c691c5-3ee8-40c2-b617-e3e5510b12b6"
        self.CALENDAR_URL = "http://172.17.0.1:8001/api/v1/"

    def get_event(self, pk):
        try:
            request = requests.get(self.CALENDAR_URL + self.CLIENT_ID + '/' +
                                   user_id + '/events/' + pk + '/')
            return request.json()
        except requests.exceptions.RequestException as e:
            pass
        return None

    def add_event(self, token, document):
        try:
            request = requests.post(self.CALENDAR_URL + document['calendar_id'] + '/events/',
                                    headers={'Authorization': token},
                                    data=document)
            return request.json()['data']['generated_keys'][0]
        except requests.exceptions.RequestException as e:
            pass
        return None

    def update_event(self, pk, document):
        try:
            request = requests.put(self.CALENDAR_URL + self.CLIENT_ID + '/' +
                                   user_id + '/events/' + pk + '/',
                                   data=document)
            return True
        except requests.exceptions.RequestException as e:
            pass
        return False

    # def delete_event(self, pk):
    #     try:
    #         request = requests.delete(self.CALENDAR_URL + self.CLIENT_ID + '/' + user_id + '/events/' + pk + '/')
    #         return True
    #     except requests.exceptions.RequestException as e:
    #         pass
    #     return False

    def free_busy(self, user_id, from_date, to_date):
        try:
            request = requests.get(self.CALENDAR_URL + 'c/' + self.CLIENT_ID + '/u/' +
                                   str(user_id) + '/events/free_busy/')
            return request.json()
        except requests.exceptions.RequestException as e:
            pass
        return []

    def list_all_events(self, user_id):
        try:
            request = requests.get(self.CALENDAR_URL + 'c/' + self.CLIENT_ID + '/u/' +
                                   str(user_id) + '/events/events/')
            return request.json()
        except requests.exceptions.RequestException as e:
            pass
        return []

    def available(self, user_id, from_date, to_date):
        try:
            request = requests.get(self.CALENDAR_URL + 'c/' + self.CLIENT_ID + '/u/' +
                                   str(user_id) + '/events/qavailable/')
            return request.json()['available']
        except requests.exceptions.RequestException as e:
            pass
        return False

    def exists(self, user_id, event_type, from_date, to_date):
        pass
