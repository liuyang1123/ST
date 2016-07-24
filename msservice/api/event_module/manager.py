import rethinkdb as r
from msservice.settings import RDB_HOST, RDB_PORT
from rethinkdb.errors import RqlRuntimeError
from api.event_module.event_type import EventType


EVENT_TYPES_DICT = {
    "call": {"unique_per_day": False,
             "is_live": False},
    "appointment": {"unique_per_day": False,
                    "is_live": True},
    "breakfast": {"unique_per_day": True,
                  "is_live": True},
    "lunch": {"unique_per_day": True,
              "is_live": True},
    "dinner": {"unique_per_day": True,
               "is_live": True},
    "hangout": {"unique_per_day": False,
                "is_live": True},
    "travel": {"unique_per_day": False,
               "is_live": True},
    "meeting": {"unique_per_day": False,
                "is_live": True}
}

TRAIN_EVENT_TYPES_DICT = {
    "call": 0,
    "appointment": 1,
    "breakfast": 2,
    "lunch": 3,
    "dinner": 4,
    "hangout": 5,
    "travel": 6,
    "meeting": 7
}

DEFAULT_EVENT_TYPE = "meeting"

EVENT_TYPES = ["call", "appointment", "breakfast", "lunch",
               "dinner", "hangout", "travel", "meeting"]

# RETHINKDB SETTINGS
EVENT_TYPES_DB = "eventtypes"
EVENT_TYPES_TABLE = "eventtypes"


class EventTypeManager:
    """
    === Preference Attributes ===
    id : UUID
    event_type : String
    is_live : Boolean
    unique_per_day : Boolean
    """

    def __init__(self):
        self.connection = r.connect(host=RDB_HOST, port=RDB_PORT)
        self.event_types_table = r.db(EVENT_TYPES_DB).table(EVENT_TYPES_TABLE)

    def close(self):
        self.connection.close()

    def list(self):
        selection = list(self.event_types_table.run(self.connection))

        return selection

    def get(self, event_type):
        selection = list(self.event_types_table.filter(
            {"event_type": event_type}).run(self.connection))
        return selection

    def event_type_object(self, event_type):
        obj = self.get(event_type)
        event_type_obj = EventType(event_type,
                                   obj["is_live"],
                                   obj["unique_per_day"])

        return event_type_obj

    def update(self, pk, document):
        updated = self.event_types_table.get(pk).replace(
            document).run(self.connection)
        return updated

    def insert(self, document):
        inserted = self.event_types_table.insert(document).run(self.connection)
        return inserted

    def delete(self, pk):
        deleted = self.event_types_table.get(pk).delete().run(self.connection)
        return deleted

    def delete_all(self):
        deleted = self.event_types_table.delete().run(self.connection)
        return deleted


LOCATION_DB = "locations"
LOCATION_TABLE = "locations"


class LocationManager:
    """
    === Preference Attributes ===
    id : UUID
    name : String
    description : String
    capacity : int
    location : String - Coordinates
    user : []
    """

    def __init__(self):
        self.connection = r.connect(host=RDB_HOST, port=RDB_PORT)
        self.locations_table = r.db(LOCATION_DB).table(LOCATION_TABLE)

    def close(self):
        self.connection.close()

    def list(self):
        selection = list(self.locations_table.run(self.connection))

        return selection

    def get(self, event_type):
        selection = list(self.locations_table.filter(
            {"event_type": event_type}).run(self.connection))
        return selection

    def location_object(self, event_type):
        obj = self.get(event_type)
        locatin_obj = Location(obj["name"],
                               obj["description"],
                               obj["capacity"],
                               obj["location"])

        return locatin_obj

    def update(self, pk, document):
        updated = self.locations_table.get(pk).replace(
            document).run(self.connection)
        return updated

    def insert(self, document):
        inserted = self.locations_table.insert(document).run(self.connection)
        return inserted

    def delete(self, pk):
        deleted = self.locations_table.get(pk).delete().run(self.connection)
        return deleted

    def delete_all(self):
        deleted = self.locations_table.delete().run(self.connection)
        return deleted


DEFAULT_EVENT_TYPES = [
    {
        "event_type": "call",
        "unique_per_day": False,
        "is_live": False
    },
    {
        "event_type": "appointment",
        "unique_per_day": False,
        "is_live": True
    },
    {
        "event_type": "breakfast",
        "unique_per_day": True,
        "is_live": True
    },
    {
        "event_type": "lunch",
        "unique_per_day": True,
        "is_live": True
    },
    {
        "event_type": "dinner",
        "unique_per_day": True,
        "is_live": True
    },
    {
        "event_type": "hangout",
        "unique_per_day": False,
        "is_live": True
    },
    {
        "event_type": "travel",
        "unique_per_day": False,
        "is_live": True
    },
    {
        "event_type": "meeting",
        "unique_per_day": False,
        "is_live": True
    }
]


def run_setup():
    connection = r.connect(host=RDB_HOST, port=RDB_PORT)
    try:
        r.db_create(EVENT_TYPES_DB).run(connection)
    except RqlRuntimeError:
        pass
    try:
        r.db(EVENT_TYPES_DB).table_create(EVENT_TYPES_TABLE).run(connection)
        r.db(LOCATION_DB).table_create(LOCATION_TABLE).run(connection)

        event_type_manager = EventTypeManager()
        for elem in DEFAULT_EVENT_TYPES:
            data = elem
            event_type_manager.insert(data)
        event_type_manager.close()
    except RqlRuntimeError:
        pass
    finally:
        connection.close()
