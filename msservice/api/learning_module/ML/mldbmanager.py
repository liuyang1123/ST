import rethinkdb as r
from rethinkdb.errors import RqlRuntimeError
from msservice.settings import RDB_HOST, RDB_PORT

# RETHINKDB SETTINGS
ML_NETWORKS_DB = "MLDB"
ML_NETWORKS_TABLE = "MLTable"

def run_setup():
    connection = r.connect(host=RDB_HOST, port=RDB_PORT)
    try:
        r.db_create(ML_NETWORKS_DB).run(connection)
    except RqlRuntimeError:
        pass
    try:
        r.db(ML_NETWORKS_DB).table_create(
            ML_NETWORKS_TABLE).run(connection)
        # Create here the default data
    except RqlRuntimeError:
        pass
    finally:
        connection.close()

class MLNetworksDBManager:
    def __init__(self):
        self.connection = r.connect(host=RDB_HOST, port=RDB_PORT)
        self.ml_table = r.db(
            ML_NETWORKS_DB).table(ML_NETWORKS_TABLE)

    def retrieve(self, instance_id, network_type):
        selection = list(
            self.ml_table.filter(
                {"instance_id": instance_id,
                 "network_type": network_type}).run(self.connection))

        if len(selection) > 0:
            return selection[0]

        return None

    def insert(self, instance_id, network_type, args):
        inserted = self.ml_table.insert({
            'instance_id': instance_id,
            'network_type': network_type,
            'args': args
        }).run(self.connection)

        return self.retrieve(instance_id, network_type)

    def update(self, instance_id, network_type, args):
        updated = self.ml_table.filter(
            {'instance_id': instance_id,
             'network_type': network_type}).replace(
                 {'instance_id': instance_id,
                  'network_type': network_type,
                  'args': args}).run(self.connection)
        return updated

    def exists(self, instance_id, network_type):
        selection = list(self.ml_table.filter(
            {"instance_id": instance_id,
             "network_type": network_type}).run(self.connection))
        if len(selection) > 0:
            return True
        return False

    def delete(self, instance_id, network_type):
        deleted = self.ml_table.filter({
            "instance_id": instance_id,
            "network_type": network_type
        }).delete().run(self.connection)
        return deleted

    def close(self):
        try:
            self.connection.close()
        except:
            pass
