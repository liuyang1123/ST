import rethinkdb as r
from rethinkdb.errors import RqlRuntimeError
from msservice.settings import RDB_HOST, RDB_PORT

# RETHINKDB SETTINGS
SOFT_CONSTRAINTS_DB = "soft_constraints"
SOFT_CONSTRAINTS_TABLE = "soft_constraints"


def run_setup():
    connection = r.connect(host=RDB_HOST, port=RDB_PORT)
    try:
        r.db_create(SOFT_CONSTRAINTS_DB).run(connection)
    except RqlRuntimeError:
        pass
    try:
        r.db(SOFT_CONSTRAINTS_DB).table_create(
            SOFT_CONSTRAINTS_TABLE).run(connection)
    except RqlRuntimeError:
        pass
    finally:
        connection.close()


class SoftConstraintsModel:
    """
    User : UUID
    Model_type : String -> BayesianNetwork, Neural network, ...
    args : Dictionary -> States, Weigths, ...
    """
    model_type = ""

    def __init__(self, user):
        self.connection = r.connect(host=RDB_HOST, port=RDB_PORT)
        self.soft_constraints_table = r.db(
            SOFT_CONSTRAINTS_DB).table(SOFT_CONSTRAINTS_TABLE)
        self.model = None
        self.user = user
        self._changed = False

    def _retrieve(self):
        user_data = list(
            self.soft_constraints_table.filter(
                {"user": self.user,
                 "model_type": self.model_type}).run(self.connection))

        if len(user_data) == 0:
            default_data = list(
                self.soft_constraints_table.filter(
                    {"user": -1,
                     "model_type": self.model_type}).run(self.connection))

            if len(default_data) > 0:
                self._insert(default_data[0]["args"])
            else:
                data = self._create_default()
                self._insert(data, -1)
                self._insert(data)

            user_data = list(
                self.soft_constraints_table.filter(
                    {"user": self.user,
                     "model_type": self.model_type}).run(self.connection))

        return user_data[0]

    def _load(self):
        return

    def _build(self, args):
        return

    def build_model(self):
        args = self._load()
        self._build(args)

    def predict(self, args):
        return

    def score_event(self, event):
        return

    def train(self, data, labels):
        return

    def save(self):
        return

    def _create_default(self):
        return

    def _insert(self, args, user=None):
        if user is None:
            user = self.user
        inserted = self.soft_constraints_table.insert({
            'user': user,
            'model_type': self.model_type,
            'args': args
        }).run(self.connection)

        return self._get()

    def _update(self, args):
        updated = self.soft_constraints_table.filter(
            {'user': self.user,
             'model_type': self.model_type}).replace(
                 {'user': self.user,
                  'model_type': self.model_type,
                  'args': args}).run(self.connection)
        return updated

    def _get(self):
        selection = list(self.soft_constraints_table.filter(
            {"user": self.user,
             "model_type": self.model_type}).run(self.connection))
        if len(selection) > 0:
            return selection[0]
        return None

    def _exists(self):
        selection = list(self.soft_constraints_table.filter(
            {"user": self.user,
             "model_type": self.model_type}).run(self.connection))
        if len(selection) > 0:
            return True
        return False

    def delete(self):
        deleted = self.soft_constraints_table.filter({
            "user": self.user,
            "model_type": self.model_type
        }).delete().run(self.connection)
        return deleted

    def _delete_all(self):
        deleted = self.soft_constraints_table.delete().run(self.connection)
        return deleted

    def close(self):
        try:
            self.connection.close()
        except:
            pass
