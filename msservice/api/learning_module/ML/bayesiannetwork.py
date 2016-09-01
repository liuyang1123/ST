import json
from pomegranate import *
from .networkmodel import NetworkModel
from .mldbmanager import MLNetworksDBManager
from api.config import SLOT_SIZE, SLOTS_PER_HOUR


types_opt = ['breakfast', 'lunch', 'dinner', 'hangout', 'call', 'meeting']
duration_opt = [15, 30, 60, 120]
day_opt = [0, 1, 2, 3, 4, 5, 6] # Monday is 0 and Sunday is 6
place_opt = ['home', 'office']
participate_opt = [False, True]
total_number_of_timeslots = SLOTS_PER_HOUR * 24  # Slots per hour * total hours


class BayesianNetworkModel(NetworkModel):
    network_type = "BayesianNetwork"

    def train(self, dataset):
        """
        Start the learning algorithm
        """
        data, labels = dataset.train.next_batch()

        if not self.model:
            self._build()

        if len(data) > 0:
            self.model.fit(data, inertia=0.8)
            self._changed = True
            self.save()

    def save(self):
        """
        Allows to save the training results, in order to restore them for later use
        """
        if self.model is None or not self._changed:
            return

        bn_states = dict()

        for state in self.model.states:
            bn_states[state.name] = state.to_json()


        db_manager = MLNetworksDBManager()
        db_manager.update(self.instance_id, self.network_type, bn_states)
        db_manager.close()

        self._changed = False

    def predict(self, query):
        if not self.model:
            self._build()

        values = dict()
        beliefs = self.model.forward_backward(query)
        for i, b in enumerate(beliefs):
            # print(json.loads(beliefs[0].to_json())["parameters"])
            values[self.model.states[i].name] = b.to_json()
        return values

    def _load(self):
        db_manager = MLNetworksDBManager()
        json_bn = db_manager.retrieve(self.instance_id,
                                      self.network_type)
        if json_bn is None:
            data = self._default()
            db_manager.insert(-1, self.network_type, data)
            db_manager.insert(self.instance_id, self.network_type, data)
            json_bn = db_manager.retrieve(self.instance_id,
                                          self.network_type)

        db_manager.close()

        return [State.from_json(json_bn["args"]["type"]),
                State.from_json(json_bn["args"]["duration"]),
                State.from_json(json_bn["args"]["day"]),
                State.from_json(json_bn["args"]["time"]),
                State.from_json(json_bn["args"]["place"]),
                State.from_json(json_bn["args"]["participant"])]

    def _process_dataset(self, dataset):
        return

    def _build(self):
        """
        Constructs the model
        """
        args = self._load()

        self.model = BayesianNetwork('BN')
        self.model.add_states(args)
        self.model.add_transition(args[0], args[1])
        self.model.add_transition(args[0], args[2])
        self.model.add_transition(args[0], args[3])
        self.model.add_transition(args[1], args[3])
        self.model.add_transition(args[2], args[4])
        self.model.add_transition(args[2], args[5])
        self.model.add_transition(args[3], args[4])
        self.model.add_transition(args[3], args[5])
        self.model.add_transition(args[4], args[5])

        self.model.bake()

        self._changed = False

    def _default(self):
        val_types, val_duration, val_day, val_time, val_place, val_part = self._get_distribution_values()

        meeting_type = DiscreteDistribution(
            val_types
        )
        duration = ConditionalProbabilityTable(
            val_duration,
            [meeting_type]
        )
        day = ConditionalProbabilityTable(
            val_day,
            [meeting_type]
        )
        time = ConditionalProbabilityTable(
            val_time,
            [meeting_type, duration]
        )
        place = ConditionalProbabilityTable(
            val_place,
            [day, time]
        )
        participant = ConditionalProbabilityTable(
            val_part,
            [day, time, place]
        )

        # Make the states
        s0 = State(meeting_type, name="type")
        s1 = State(duration, name="duration")
        s2 = State(day, name="day")
        s3 = State(time, name="time")
        s4 = State(place, name="place")
        s5 = State(participant, name="participant")

        # args = [s0, s1, s2, s3, s4, s5]
        # self._build(args)  # The model is completed!

        bn_states = dict()
        bn_states[s0.name] = s0.to_json()
        bn_states[s1.name] = s1.to_json()
        bn_states[s2.name] = s2.to_json()
        bn_states[s3.name] = s3.to_json()
        bn_states[s4.name] = s4.to_json()
        bn_states[s5.name] = s5.to_json()

        # stype = State.from_json(bn_states["type"])
        # sdura = State.from_json(bn_states["duration"])
        # sday = State.from_json(bn_states["day"])
        # stime = State.from_json(bn_states["time"])
        # splac = State.from_json(bn_states["place"])
        # spart = State.from_json(bn_states["participant"])

        return bn_states

    def _get_distribution_values(self):
        result_types = dict()
        result_duration = list()
        result_day = list()
        result_time = list()
        result_place = list()
        result_part = list()

        for e in types_opt:
            # Event type
            result_types[e] = 1. / len(types_opt)
            # Event duration
            for dur in duration_opt:
                result_duration.append([e, dur, 1. / len(duration_opt)])

                # Event time
                for t in range(total_number_of_timeslots):
                    result_time.append([e, dur,
                                        t, 1. / total_number_of_timeslots])

                    # Event place
                    for _d in day_opt:
                        for p in place_opt:
                            result_place.append([_d, t,
                                                 p, 1. / len(place_opt)])

                            for b in participate_opt:
                                # Event participate
                                result_part.append([_d, t, p, b,
                                                    1. / len(participate_opt)])

            # Event day
            for day in day_opt:
                result_day.append([e, day, 1. / len(day_opt)])

        return result_types, result_duration, result_day, result_time, result_place, result_part


    def score_event(self, event):
        # TODO Remove this method, this should be done outside
        hour = event.start_time.time().hour
        minute = event.start_time.time().minute
        time = hour * SLOTS_PER_HOUR + minute / SLOT_SIZE

        args = {"type": str(event.event_type),
                "duration": event.duration,
                "day": event.start_time.weekday(),
                "time": time}

        result = self.predict(args)

        return float(json.loads(
            result["participant"]).get("parameters")[0]['true'])
