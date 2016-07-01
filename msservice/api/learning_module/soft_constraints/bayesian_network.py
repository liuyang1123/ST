import json
from pomegranate import *
from .soft_constraints_model import SoftConstraintsModel
from api.config import SLOT_SIZE, SLOTS_PER_HOUR


types_opt = ['breakfast', 'lunch', 'dinner',
             'hangout', 'call', 'meeting']

duration_opt = [15, 30, 60, 120]

# Monday is 0 and Sunday is 6
day_opt = [0, 1, 2, 3, 4, 5, 6]

# TODO Construct a specific BN based on the user data
place_opt = ['Home', 'Office', 'X']

participate_opt = [False, True]

total_number_of_timeslots = SLOTS_PER_HOUR * 24  # Slots per hour * total hours


class BayesianNetworkModel(SoftConstraintsModel):
    model_type = "BayesianNetwork"

    def _load(self):
        json_bn = self._retrieve()
        return [State.from_json(json_bn["args"]["type"]),
                State.from_json(json_bn["args"]["duration"]),
                State.from_json(json_bn["args"]["day"]),
                State.from_json(json_bn["args"]["time"]),
                State.from_json(json_bn["args"]["place"]),
                State.from_json(json_bn["args"]["participant"])]

    def _build(self, args):
        self.model = BayesianNetwork("BNfMS_" + str(self.user))
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

    def predict(self, args):
        if not self.model:
            self.build_model()

        values = dict()
        beliefs = self.model.forward_backward(args)
        for i, b in enumerate(beliefs):
            # print(json.loads(beliefs[0].to_json())["parameters"])
            values[self.model.states[i].name] = b.to_json()
        return values

    def score_event(self, event):
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

    def train(self, data, labels):
        if not self.model and len(data) > 0:
            self.build_model()

        if len(data) > 0:
            self.model.fit(data, inertia=0.8)
            self._changed = True
            self.save()

    def save(self):
        if not self.model or not self._changed:
            return

        bn_states = dict()

        for state in self.model.states:
            bn_states[state.name] = state.to_json()

        self._update(bn_states)

        self._changed = False

    def _create_default(self):
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


# TODO
# training_data = [
#     ['breakfast', 30, 'Saturday', '9:00', -1, True],
#     ['breakfast', 30, 'Saturday', '9:30', -1, True],
#     ['breakfast', 30, 'Saturday', '10:00', -1, True],
#     ['breakfast', 30, 'Saturday', '10:30', -1, True],
#     ['breakfast', 60, 'Saturday', '9:00', -1, True],
#     ['breakfast', 60, 'Saturday', '9:30', -1, True],
#     ['breakfast', 60, 'Saturday', '10:00', -1, True],
#     ['breakfast', 60, 'Saturday', '10:30', -1, True],
#     ['breakfast', 30, 'Sunday', '9:00', -1, True],
#     ['breakfast', 30, 'Sunday', '9:30', -1, True],
#     ['breakfast', 30, 'Sunday', '10:00', -1, True],
#     ['breakfast', 30, 'Sunday', '10:30', -1, True],
#     ['breakfast', 60, 'Sunday', '9:00', -1, True],
#     ['breakfast', 60, 'Sunday', '9:30', -1, True],
#     ['breakfast', 60, 'Sunday', '10:00', -1, True],
#     ['breakfast', 60, 'Sunday', '10:30', -1, True],
#     ['breakfast', 30, 'Monday', '8:00', -1, True],
#     ['breakfast', 30, 'Monday', '8:30', -1, True],
#     ['breakfast', 30, 'Tuesday', '8:00', -1, True],
#     ['breakfast', 30, 'Tuesday', '8:30', -1, True],
#     ['breakfast', 30, 'Wednesday', '8:00', -1, True],
#     ['breakfast', 30, 'Wednesday', '8:30', -1, True],
#     ['breakfast', 30, 'Thursday', '8:00', -1, True],
#     ['breakfast', 30, 'Thursday', '8:30', -1, True],
#     ['breakfast', 30, 'Friday', '8:00', -1, True],
#     ['breakfast', 30, 'Friday', '8:30', -1, True],
#
#     ['lunch', 60, 'Saturday', '12:00', -1, True],
#     ['lunch', 60, 'Saturday', '12:30', -1, True],
#     ['lunch', 60, 'Saturday', '13:00', -1, True],
#     ['lunch', 60, 'Saturday', '13:30', -1, True],
#     ['lunch', 60, 'Sunday', '12:00', -1, True],
#     ['lunch', 60, 'Sunday', '12:30', -1, True],
#     ['lunch', 60, 'Sunday', '13:00', -1, True],
#     ['lunch', 60, 'Sunday', '13:30', -1, True],
#     ['lunch', 60, 'Monday', '12:00', -1, True],
#     ['lunch', 60, 'Monday', '13:00', -1, True],
#     ['lunch', 60, 'Tuesday', '12:00', -1, True],
#     ['lunch', 60, 'Tuesday', '13:00', -1, True],
#     ['lunch', 60, 'Wednesday', '12:00', -1, True],
#     ['lunch', 60, 'Wednesday', '13:00', -1, True],
#     ['lunch', 60, 'Thursday', '12:00', -1, True],
#     ['lunch', 60, 'Thursday', '13:00', -1, True],
#     ['lunch', 60, 'Friday', '12:00', -1, True],
#     ['lunch', 60, 'Friday', '13:00', -1, True],
#
#     ['dinner', 60, 'Saturday', '20:00', -1, True],
#     ['dinner', 60, 'Saturday', '21:00', -1, True],
#     ['dinner', 60, 'Saturday', '22:00', -1, True],
#     ['dinner', 60, 'Saturday', '21:30', -1, True],
#     ['dinner', 60, 'Sunday', '20:00', -1, True],
#     ['dinner', 60, 'Sunday', '21:00', -1, True],
#     ['dinner', 60, 'Sunday', '22:00', -1, True],
#     ['dinner', 60, 'Sunday', '21:30', -1, True],
#     ['dinner', 60, 'Monday', '20:00', -1, True],
#     ['dinner', 60, 'Monday', '21:00', -1, True],
#     ['dinner', 60, 'Tuesday', '20:00', -1, True],
#     ['dinner', 60, 'Tuesday', '21:00', -1, True],
#     ['dinner', 60, 'Wednesday', '20:00', -1, True],
#     ['dinner', 60, 'Wednesday', '21:00', -1, True],
#     ['dinner', 60, 'Thursday', '20:00', -1, True],
#     ['dinner', 60, 'Thursday', '21:00', -1, True],
#     ['dinner', 60, 'Friday', '20:00', -1, True],
#     ['dinner', 60, 'Friday', '22:00', -1, True],
#
#     ['hangout', 60, 'Saturday', '14:00', -1, True],
#     ['hangout', 60, 'Saturday', '15:00', -1, True],
#     ['hangout', 60, 'Saturday', '16:00', -1, True],
#     ['hangout', 60, 'Saturday', '18:30', -1, True],
#     ['hangout', 60, 'Sunday', '15:00', -1, True],
#     ['hangout', 60, 'Sunday', '16:00', -1, True],
#     ['hangout', 60, 'Sunday', '17:00', -1, True],
#     ['hangout', 60, 'Sunday', '16:30', -1, True],
#
#     # Calls only in the morning
#     ['call', 15, 'Monday', '9:00', -1, True],
#     ['call', 15, 'Monday', '9:30', -1, True],
#     ['call', 15, 'Monday', '10:00', -1, True],
#     ['call', 15, 'Monday', '10:30', -1, True],
#     ['call', 30, 'Monday', '9:00', -1, True],
#     ['call', 30, 'Monday', '9:30', -1, True],
#     ['call', 30, 'Monday', '10:00', -1, True],
#     ['call', 30, 'Monday', '10:30', -1, True],
#     ['call', 15, 'Tuesday', '9:00', -1, True],
#     ['call', 15, 'Tuesday', '9:30', -1, True],
#     ['call', 15, 'Tuesday', '10:00', -1, True],
#     ['call', 15, 'Tuesday', '10:30', -1, True],
#     ['call', 30, 'Tuesday', '9:00', -1, True],
#     ['call', 30, 'Tuesday', '9:30', -1, True],
#     ['call', 30, 'Tuesday', '10:00', -1, True],
#     ['call', 30, 'Tuesday', '10:30', -1, True],
#     ['call', 15, 'Wednesday', '9:00', -1, True],
#     ['call', 15, 'Wednesday', '9:30', -1, True],
#     ['call', 15, 'Wednesday', '10:00', -1, True],
#     ['call', 15, 'Wednesday', '10:30', -1, True],
#     ['call', 30, 'Wednesday', '9:00', -1, True],
#     ['call', 30, 'Wednesday', '9:30', -1, True],
#     ['call', 30, 'Wednesday', '10:00', -1, True],
#     ['call', 30, 'Wednesday', '10:30', -1, True],
#     ['call', 15, 'Thursday', '9:00', -1, True],
#     ['call', 15, 'Thursday', '9:30', -1, True],
#     ['call', 15, 'Thursday', '10:00', -1, True],
#     ['call', 15, 'Thursday', '10:30', -1, True],
#     ['call', 30, 'Thursday', '9:00', -1, True],
#     ['call', 30, 'Thursday', '9:30', -1, True],
#     ['call', 30, 'Thursday', '10:00', -1, True],
#     ['call', 30, 'Thursday', '10:30', -1, True],
#     ['call', 15, 'Friday', '9:00', -1, True],
#     ['call', 15, 'Friday', '9:30', -1, True],
#     ['call', 15, 'Friday', '10:00', -1, True],
#     ['call', 15, 'Friday', '10:30', -1, True],
#     ['call', 30, 'Friday', '9:00', -1, True],
#     ['call', 30, 'Friday', '9:30', -1, True],
#     ['call', 30, 'Friday', '10:00', -1, True],
#     ['call', 30, 'Friday', '10:30', -1, True],
#
#     ['meeting', 15, 'Monday', '9:00', -1, True],
#     ['meeting', 15, 'Monday', '9:30', -1, True],
#     ['meeting', 15, 'Monday', '10:00', -1, True],
#     ['meeting', 15, 'Monday', '10:30', -1, True],
#     ['meeting', 30, 'Monday', '9:00', -1, True],
#     ['meeting', 30, 'Monday', '9:30', -1, True],
#     ['meeting', 30, 'Monday', '10:00', -1, True],
#     ['meeting', 30, 'Monday', '10:30', -1, True],
#     ['meeting', 15, 'Tuesday', '9:00', -1, True],
#     ['meeting', 15, 'Tuesday', '9:30', -1, True],
#     ['meeting', 15, 'Tuesday', '10:00', -1, True],
#     ['meeting', 15, 'Tuesday', '10:30', -1, True],
#     ['meeting', 30, 'Tuesday', '9:00', -1, True],
#     ['meeting', 30, 'Tuesday', '9:30', -1, True],
#     ['meeting', 30, 'Tuesday', '10:00', -1, True],
#     ['meeting', 30, 'Tuesday', '10:30', -1, True],
#     ['meeting', 15, 'Wednesday', '9:00', -1, True],
#     ['meeting', 15, 'Wednesday', '9:30', -1, True],
#     ['meeting', 15, 'Wednesday', '10:00', -1, True],
#     ['meeting', 15, 'Wednesday', '10:30', -1, True],
#     ['meeting', 30, 'Wednesday', '9:00', -1, True],
#     ['meeting', 30, 'Wednesday', '9:30', -1, True],
#     ['meeting', 30, 'Wednesday', '10:00', -1, True],
#     ['meeting', 30, 'Wednesday', '10:30', -1, True],
#     ['meeting', 15, 'Thursday', '9:00', -1, True],
#     ['meeting', 15, 'Thursday', '9:30', -1, True],
#     ['meeting', 15, 'Thursday', '10:00', -1, True],
#     ['meeting', 15, 'Thursday', '10:30', -1, True],
#     ['meeting', 30, 'Thursday', '9:00', -1, True],
#     ['meeting', 30, 'Thursday', '9:30', -1, True],
#     ['meeting', 30, 'Thursday', '10:00', -1, True],
#     ['meeting', 30, 'Thursday', '10:30', -1, True],
#     ['meeting', 15, 'Friday', '9:00', -1, True],
#     ['meeting', 15, 'Friday', '9:30', -1, True],
#     ['meeting', 15, 'Friday', '10:00', -1, True],
#     ['meeting', 15, 'Friday', '10:30', -1, True],
#     ['meeting', 30, 'Friday', '9:00', -1, True],
#     ['meeting', 30, 'Friday', '9:30', -1, True],
#     ['meeting', 30, 'Friday', '10:00', -1, True],
#     ['meeting', 30, 'Friday', '10:30', -1, True],
#     ['meeting', 60, 'Friday', '14:00', -1, True],
# ]
# # self.model.fit(training_data)
