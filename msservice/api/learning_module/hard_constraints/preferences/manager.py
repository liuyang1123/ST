import rethinkdb as r
import ast
import json
from msservice.settings import RDB_HOST, RDB_PORT
from rethinkdb.errors import RqlRuntimeError
from api.learning_module.hard_constraints.preferences import preference
from api.event_module.manager import EVENT_TYPES_DICT, DEFAULT_EVENT_TYPE
from api.event_module.event_type import EventType

USER_PREFERENCES = {
    "BookableHoursPreference": preference.BookableHoursPreference,
    "DoNotDisturbPreference": preference.DoNotDisturbPreference,
    "DurationPreference": preference.DurationPreference,
    "TimeBetweenPreference": preference.TimeBetweenPreference,
    "MaxDistancePreference": preference.MaxDistancePreference,
    "ModeOfCommunicationPreference": preference.ModeOfCommunicationPreference
}

# RETHINKDB SETTINGS
USER_PREFERENCES_DB = "user_preferences"
USER_PREFERENCES_TABLE = "user_preferences"


class UserPreferencesManager:
    """
    === Preference Attributes ===
    id : UUID
    user_id : UUID
    event_type : String
    preference_type : String
    args : Dictionary
    """

    def __init__(self):
        self.connection = r.connect(host=RDB_HOST, port=RDB_PORT)
        self.USER_PREFERENCES_TABLE = r.db(
            USER_PREFERENCES_DB).table(USER_PREFERENCES_TABLE)

    def close(self):
        self.connection.close()

    def list(self, user_id):
        selection = list(self.USER_PREFERENCES_TABLE.filter(
            {"user_id": user_id}).run(self.connection))

        return selection

    def get(self, user_id, pk):
        return list(self.USER_PREFERENCES_TABLE.filter({
            "id": pk, "user_id": user_id
        }).run(self.connection))

    def get_all(self, user_id):
        selection = list(self.USER_PREFERENCES_TABLE.filter(
            {"user_id": user_id}).run(self.connection))
        return selection

    def get_list_objects(self, user_id):
        objs = self.list(user_id)

        preferences = []
        for o in objs:
            pref = USER_PREFERENCES.get(o["preference_type"])

            preferences.append(
                pref(preference_name=o.get("preference_name", ""),
                     event_type=o.get("event_type", ""),
                     args=ast.literal_eval(o.get("attributes", {})))
                )

        return preferences

    def update(self, pk, document):
        updated = self.USER_PREFERENCES_TABLE.get(pk).replace(
            document).run(self.connection)
        return updated

    def insert(self, document):
        inserted = self.USER_PREFERENCES_TABLE.insert(
            document).run(self.connection)
        return inserted

    def delete(self, user_id, pk):
        deleted = self.USER_PREFERENCES_TABLE.filter({
            "id": pk, "user_id": user_id
        }).delete().run(self.connection)
        return deleted

    def delete_all(self, user_id):
        return self.USER_PREFERENCES_TABLE.filter({
            "user_id": user_id
        }).delete().run(self.connection)

DEFAULT_USER_PREFERENCES = [
    {
        "event_type": "",
        "user_id": "",
        "preference_type": "ANY",
        "args": {}
    },
]


def run_setup():
    connection = r.connect(host=RDB_HOST, port=RDB_PORT)
    try:
        r.db_create(USER_PREFERENCES_DB).run(connection)
    except RqlRuntimeError:
        pass
    try:
        r.db(USER_PREFERENCES_DB).table_create(
            USER_PREFERENCES_TABLE).run(connection)

        # preferences_manager = UserPreferencesManager()
        # for elem in DEFAULT_USER_PREFERENCES:
        #     data = elem
        #     preferences_manager.insert(data)
        # preferences_manager.close()
    except RqlRuntimeError:
        pass
    finally:
        connection.close()
