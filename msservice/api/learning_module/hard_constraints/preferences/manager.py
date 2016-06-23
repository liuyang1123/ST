import rethinkdb as r
from rethinkdb.errors import RqlRuntimeError
from api.learning_module.hard_constraints.preferences import preference

USER_PREFERENCES = ["BookableHoursPreference", "DoNotDisturbPreference",
                    "DurationPreference", "TimeBetweenPreference",
                    "MaxDistancePreference", "ModeOfCommunicationPreference"]

# RETHINKDB SETTINGS
RDB_HOST = "rethinkdb"
RDB_PORT = 28015
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

    def list(self):
        selection = list(self.USER_PREFERENCES_TABLE.run(self.connection))

        return selection

    def get(self, user_id):
        selection = list(self.USER_PREFERENCES_TABLE.filter(
            {"user_id": user_id}).run(self.connection))
        return selection

    def get_list_objects(self, user_id):
        objs = self.get(user_id)
        event_type_obj = EventType(event_type,
                                   obj["is_live"],
                                   obj["unique_per_day"])
        preferences = []
        for o in objs:
            if o["preference_type"] == "BookableHoursPreference":
                pref = preference.BookableHoursPreference(o["preference_type"],
                                                          event_type,
                                                          o["args"])

            preferences.append(pref)
        # if "DoNotDisturbPreference",
        #                     "DurationPreference", "TimeBetweenPreference",
        #                     "MaxDistancePreference", "ModeOfCommunicationPreference"

        return preferences

    def update(self, pk, document):
        updated = self.USER_PREFERENCES_TABLE.get(pk).replace(
            document).run(self.connection)
        return updated

    def insert(self, document):
        inserted = self.USER_PREFERENCES_TABLE.insert(
            document).run(self.connection)
        return inserted

    def delete(self, pk):
        deleted = self.USER_PREFERENCES_TABLE.get(
            pk).delete().run(self.connection)
        return deleted

    def delete_all(self):
        deleted = self.USER_PREFERENCES_TABLE.delete().run(self.connection)
        return deleted


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
