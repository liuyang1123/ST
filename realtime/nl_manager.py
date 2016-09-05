"""
Responsabilities of this module:
    - Keep track of wit.ai conversation (chat) and store them.
        - onMessage call wit.ai, and return the response.
    - Manage the connection with rethinkdb.
    - Save the requests in natural language of the users.
    - Allow filtering in changefeeds based on session.
"""

import rethinkdb as r
from rethinkdb.errors import RqlRuntimeError

# RETHINKDB SETTINGS
RDB_HOST = "127.0.0.1"
RDB_PORT = 28015
NL_DB = "nl"
CONVERSATION_TABLE = "chat"


class ConversationManager:
    """
    === Preference Attributes ===
    id : UUID
    session : UUID
    user : int
    message : String
    attributes : dict
    """

    def __init__(self):
        self.connection = r.connect(host=RDB_HOST, port=RDB_PORT)
        self.conversation_table = r.db(NL_DB).table(CONVERSATION_TABLE)

    def close(self):
        self.connection.close()

    def list(self, session):
        selection = list(self.conversation_table.filter({
            "session": session
        }).run(self.connection))

        return selection

    def insert(self, document):
        inserted = self.conversation_table.insert(
            document).run(self.connection)

        return inserted

    def delete(self, pk):
        deleted = self.conversation_table.get(pk).delete().run(self.connection)

        return deleted

    def delete_all(self):
        deleted = self.conversation_table.delete().run(self.connection)

        return deleted


def run_setup():
    connection = r.connect(host=RDB_HOST, port=RDB_PORT)
    try:
        r.db_create(NL_DB).run(connection)
    except RqlRuntimeError:
        pass
    try:
        r.db(NL_DB).table_create(CONVERSATION_TABLE).run(connection)
    except RqlRuntimeError:
        pass
    finally:
        connection.close()
# Change this as the other services
run_setup()
