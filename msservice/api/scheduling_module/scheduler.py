from api.learning_module.hard_constraints.rules.manager import RulesManager
from api.learning_module.hard_constraints.preferences.manager import UserPreferencesManager
from api.learning_module.soft_constraints.bayesian_network import BayesianNetworkModel


class Scheduler:

    def __init__(self, list_of_events, sc_model=BayesianNetworkModel):
        # TODO Sort self.events by its importance and precedence
        self.events = list_of_events  # self._sort_events()
        # upm = UserPreferencesManager()
        # preferences = ump.get_list_objects()
        # self.hc = HardConstraint()
        self.sc = sc_model()

    def _sort_events(self):
        """
        Sorts the events by importance, keeping the corresponding precedence.
        """
        return

    def _get_available_slots(self, event):
        """
        Call the database manager
        """
        return

    def _get_attendees_with_an_account(self, attendees):
        result = []
        # Returns a list of user_id
        for attendee in attendees:
            try:
                request = requests.post("http://172.17.0.1:8000/api/v1/" + CLIENT_ID + '/get_user_by_email/',
                                        data={"email": attendee})
                if request.status_code == 200:
                    result.append(request.json()['user_id'])
            except requests.exceptions.RequestException as e:
                pass
        return result

    def best_slots(self, k=1, alpha=1, beta=1):
        """
        Calculate the objective function for all the meetings, and return the
        best time for each one of them.

        TODO:
            - Hasta ahora se prioriza los eventos y por cada evento se trata de elegir la mejor opcion.
                - La idea seria probar distintas opciones que maximice el beneficio general y no solo
                por evento. Ver Meeting Scheduling Based on Swarm Intelligence
        """
        for i, event in enumerate(events):
            # TODO (1) Figure out if it's possible or better to make a query that selects all the corresponding slots for all users at the same time
            # TODO (2) Enable re-scheduling
            # TODO (3) Make more efficient the get_available_slots so that it doesn't query the database if it's is in the same range of time
            # TODO (4) Recommend a place if not one is given. If we are in an
            # office context, that location should be one of the free rooms.
            # Look for yelp. And historic data.
            available_slots = self._get_available_slots(
                event)  # This are the initiators timeslots
            slots = {}
            for j, slot in enumerate(available_slots):
                event.start_time = slot.get_initial()
                event.end_time = slot.get_end()
                if self.hc.score(event):
                    sc_score = self.sc.predict(event)
                    actual_pref_score = 0.0
                    for p in event.participants:
                        actual_pref_score += p.get_preference_score(event)
                    slots[j] = alpha * sc_score + beta * actual_pref_score

            ids_sorted_by_constraints = sorted(slots,
                                               key=slots.get)[:k]  # Sort and get the top k slots
            self.save_selected_slots(
                event, [available_slots[r] for r in ids_sorted_by_constraints])
            # This should be a parameter, if I want to see different options, or to just schedule
            # Remove the selected slot when implementing (3)
            # The top option
            self.event.assign(available_slots[ids_sorted_by_constraints[0]])

        def select_best_slot(self, k=1, alpha=1, beta=1):
            self.best_slots()

# S = slots(day=n)
# for a in attendees:
#     pA = scores(S)
# find_top_intersection(pA)


# # Una implementacion adaptada de lo presentado en: "A distributed multi-agent meeting scheduler"
# def event_scheduler(scheduling_task, event): # Returns a member of SCHEDULE_STATUSES
#     """
#
#     """
#
#     initiator_id = scheduling_task['initiator_id']
#     start = event.get('start', now()) # TODO Add parameter timezone
#     end = event.get('end', now() + timedelta(days=14)) # TODO Add parameter timezone
#
#     pref_manager = PreferenceManager()
#     duration = event.get('duration', pref_manager.get_one_preference(initiator_id, 'durationpreference', event['categories'][0]))
#     pref_manager.close()
#
#     initiators_time_slot_manager = TimeSlotManager(initiator_id,
#                                                    parse(start),
#                                                    parse(end),
#                                                    duration)
#
#     successful_negotation = False # Negotation state
#     everyone_agrees = 0
#     calendar_client = CalendarDBClient()
#
#     attendees_with_an_account = get_attendees_with_an_account(event['attendees'])
#     attendees_with_an_account.append(initiator_id)
#     number_of_attendees = len(attendees_with_an_account)
#
#     event_type_rules = EventTypeManager().event_type_object(event['categories'][0]).get_rules()
#
#     preference_manager = PreferenceManager()
#     while not successful_negotation and initiators_time_slot_manager.has_next():
#         everyone_agrees = 0
#         time_slot = initiators_time_slot_manager.next()
#         # Si por cada attendee, el evento cumple con las reglas y HC(attendee) > min_confidence_score
#         #    everyone_agrees += 1
#         #for attendee in attendees_with_an_account:
#         #    if HC.is_valid(attendee)
#         #        everyone_agrees += 1
#         for rule in event_type_rules:
#             if rule.is_valid(event, attendees_with_an_account, time_slot): # and HC.xxx
#                 everyone_agrees += 1
#             else:
#                 everyone_agrees = 0
#                 break
#         if everyone_agrees > 0:
#             for attendee in attendees_with_an_account:
#                 preference_obj = preference_manager.preference_object(attendee)
#                 if preference_obj.confidence_score(event, time_slot) < 1: # MIN_DELTA = 1
#                     everyone_agrees -= 1
#
#         if everyone_agrees == number_of_attendees:
#             successful_negotation = True
#             # Save results
#             # Send notifications(attendees_with_an_account, event['attendees'])
#             # Wait for confirmation
#             #   On confirmation: client.update_calendar(event, time_slot)
