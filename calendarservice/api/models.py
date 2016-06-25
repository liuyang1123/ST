import rethinkdb as r
from rethinkdb.errors import RqlRuntimeError

# CHOICES
UNKNOWN = "unknown"
PROVIDERS = ["apple", "exchange", "google", "live_connect", "office365"]
PARTICIPATION_STATUS = [
    UNKNOWN,
    "needs_action",
    "accepted",
    "declined",
    "tentative"]
TRANSPARENCY = [UNKNOWN, "opaque", "transparent"]
EVENT_STATUS = [UNKNOWN, "tentative", "confirmed", "cancelled"]

# RETHINKDB SETTINGS
RDB_HOST = "rethinkdb"
RDB_PORT = 28015
CALENDAR_DB = "calendarapi"
CALENDAR_TABLE = "calendars"
EVENT_TABLE = "events"


def run_setup():
    connection = r.connect(host=RDB_HOST, port=RDB_PORT)
    try:
        r.db_create(CALENDAR_DB).run(connection)
    except RqlRuntimeError:
        pass
    try:
        r.db(CALENDAR_DB).table_create(CALENDAR_TABLE).run(connection)
    except RqlRuntimeError:
        pass
    try:
        r.db(CALENDAR_DB).table_create(EVENT_TABLE).run(connection)
    except RqlRuntimeError:
        pass
    finally:
        connection.close()


class Calendar:
    """
    === Calendar Attributes ===
    calendar_id : UUID
    user_id : UUID
    provider_name : String
        # ex. apple, cronofy, exchange, google, live_connect, office365
    # profile_name : String
    #     example@cronofy.com
    calendar_name : String
        # ex. "Home", "Work"
    calendar_readonly : Boolean
    calendar_deleted : Boolean
    """

    def __init__(self):
        self.connection = r.connect(host=RDB_HOST, port=RDB_PORT)
        self.calendar_table = r.db(CALENDAR_DB).table(CALENDAR_TABLE)

    def list(self, user_id):
        selection = list(self.calendar_table.filter(
            {"user_id": user_id}).run(self.connection))

        return selection

    def list_all(self):
        selection = list(self.calendar_table.run(self.connection))
        return selection

    def get(self, user_id, pk):
        selection = list(self.calendar_table.filter({"id": pk,
                                                     "user_id": user_id}
                                                    ).run(self.connection))

        return selection

    def update(self, user_id, pk, document):
        element = self.calendar_table.get(pk).run(self.connection)
        if element['user_id'] == user_id:
            updated = self.calendar_table.get(
                pk).update(document).run(self.connection)
            return updated
        return None

    def insert(self, document):
        inserted = self.calendar_table.insert(document).run(self.connection)
        return inserted

    def delete(self, user_id, pk):
        element = self.calendar_table.get(pk).run(self.connection)
        if element['user_id'] == user_id:
            deleted = self.calendar_table.get(pk).delete().run(self.connection)
            return deleted
        return None

    def close(self):
        self.connection.close()

    def is_valid_provider_name(self, name):
        if name.lower() in PROVIDERS:
            return True
        return False


class Event:
    """
    === Calendar Attributes ===
    calendar_id : UUID
    user_id : UUID
    event_id : UUID
    summary : String
        # ex. Board meeting
    description : String
        # ex. Discuss plans for the next quarter.
    deleted : Boolean
    start : DateTime ISO8601
    end : DateTime ISO8601
    duration : Integer
    location : Location (JSON)
        # ex. {"description": "Office"}
    participation_status : String
        # ex. needs_action accepted declined tentative unknown
    transparency : String
        # ex. opaque transparent unknown
    event_status : String
        # ex. tentative confirmed cancelled unknown
    categories : List of strings
        # ex. ["Lunch", "Office", "Starbucks"] - Free text
    attendees : List of dictionaries
        # ex. [{"email": "", "display_name": "", "status": "needs_action"}]
    is_fixed : Boolean
    created : DateTime ISO8601
    updated : DateTime ISO8601
    """

    def __init__(self):
        self.connection = r.connect(host=RDB_HOST, port=RDB_PORT)
        self.event_table = r.db(CALENDAR_DB).table(EVENT_TABLE)

    def is_valid_participation_status(self, status):
        if status.lower() in PARTICIPATION_STATUS:
            return True
        return False

    def is_valid_transparency(self, transparency):
        if transparency.lower() in TRANSPARENCY:
            return True
        return False

    def is_valid_event_status(self, status):
        if status.lower() in EVENT_STATUS:
            return True
        return False

    def close(self):
        self.connection.close()

    def get_event(self, user_id, calendar_id, event_id):
        selection = list(self.event_table.filter({
            "user_id": user_id, "id": event_id, "calendar_id": calendar_id}
        ).run(self.connection))

        return selection

    def get_event_with_pk(self, event_id):
        selection = list(self.event_table.filter(
            {"id": event_id}
        ).run(self.connection))

        return selection

    def update_event(self, calendar_id, pk, document):
        element = self.event_table.get(pk).run(self.connection)
        if element['calendar_id'] == calendar_id:
            updated = self.event_table.get(pk).update(
                document).run(self.connection)
            return updated
        return None

    def update_event_with_pk(self, pk, document):
        updated = self.event_table.get(pk).update(
            document).run(self.connection)
        return updated

    def insert_event(self, document):
        inserted = self.event_table.insert(document).run(self.connection)
        return inserted

    def delete_event(self, calendar_id, pk):
        element = self.event_table.get(pk).run(self.connection)
        if element['calendar_id'] == calendar_id:
            deleted = self.event_table.get(pk).delete().run(self.connection)
            return deleted
        return None

    def list_events(self, user_id, calendar_id):
        selection = list(self.event_table.filter(
            {"user_id": user_id, "calendar_id": calendar_id}).run(self.connection))

        return selection

    def list_all_events(self, user_id):
        selection = list(self.event_table.filter(
            {"user_id": user_id}).run(self.connection))

        return selection

    def list_all_events_for_training(self, user_id):
        selection = list(self.event_table.filter(
            {"user_id": user_id}).filter(
                r.row.has_fields('start') &
                r.row.has_fields('end')).run(self.connection))

        return selection

    def free_busy(self, user_id, from_date, to_date):
        dtz = '+00:00'  # Default Timezone

        selection = list(self.event_table.filter({"user_id": user_id}).filter(
            r.iso8601(r.row['start'],
                      default_timezone=dtz).during(
                          r.iso8601(from_date,
                                    default_timezone=dtz),
                          r.iso8601(to_date,
                                    default_timezone=dtz),
                          left_bound="closed",
                          right_bound="closed") |
            r.iso8601(r.row['end'],
                      default_timezone=dtz).during(
                          r.iso8601(from_date,
                                    default_timezone=dtz),
                          r.iso8601(to_date,
                                    default_timezone=dtz),
                          left_bound="closed",
                          right_bound="closed")
        ).run(self.connection))

        return selection

    def available(self, user_id, from_date, to_date):
        if len(self.free_busy(user_id, from_date, to_date)) == 0:
            return True
        return False
