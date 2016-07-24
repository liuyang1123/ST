from api.learning_module.hard_constraints.rules.manager import RulesManager
from api.event_module.timeslots import TimeSlotManager
from api.event_module.event import Event
from api.models import Invitation
from api.event_module.calendar_client import CalendarDBClient

# TODO Add the specific timezone to the datetimes. Right now is empty.


class Scheduler:

    def __init__(self, list_of_events, tasks=None):
        self.events = self._sort_events(list_of_events)
        self.tasks = tasks

    def _sort_events(self, events_to_sort):
        """
        Sorts the events by importance, keeping the corresponding precedence.
        """
        # TODO Sort self.events by its importance and precedence
        return events_to_sort

    def _get_available_slots(self, event):
        """
        Call the database manager
        Returns a set of free timeslots
        """
        return []

    def best_slots(self, event, k=1, alpha=1, beta=1, invalid=None):
        """
        Calculate the objective function for all the meetings, and return the k
        best times for each one of them.

        TODO:
            - Hasta ahora se prioriza los eventos y por cada evento se trata de elegir la mejor opcion.
                - La idea seria probar distintas opciones que maximice el beneficio general y no solo
                por evento. Ver Meeting Scheduling Based on Swarm Intelligence

            Invalid are timeslots rejected by a user. Used in order to not choose the same options.
        """
        hc = RulesManager()

        duration = event.duration
        if duration == -1:
            duration = 30  # TODO infer
        ts_manager = TimeSlotManager(user_id=event.participants[-1].user_id,
                                     start_period=event.start_time,
                                     end_period=event.end_time,
                                     duration=duration)

        event_to_schedule = Event(participants=event.participants,
                                  event_type=event.event_type,
                                  description=event.description,
                                  duration=duration,
                                  start_time=event.start_time,
                                  end_time=event.end_time,
                                  location=event.location)
        slots = {}
        slots_score = {}
        j = 0
        rule_modifier = False
        while ts_manager.has_next():
            slot = ts_manager.next()
            if slot is None:
                break

            event_to_schedule.start_time = slot.get_start()
            event_to_schedule.end_time = slot.get_end()

            if hc.is_valid(event_to_schedule) == 1:
                pref_score = 0.0
                sc_score = 0.0
                for p in event_to_schedule.participants:
                    pref_score += p.get_score(event_to_schedule)
                    sc_score += p.get_prediction(event_to_schedule)
                    # print("Participant: " + str(p.user_id))
                    # print("Score: " + str(p.get_score(event_to_schedule)))
                    # print("Prediction: " + str(p.get_prediction(event_to_schedule)))
                slots[j] = {"start": slot.get_start(),
                            "end": slot.get_end()}
                slots_score[j] = alpha * pref_score + beta * sc_score
                j += 1
                # TODO  Break the loop.
                # Right now it's going to analyze 15 days. Infer the best day.
            else:
                if hc.has_possible_solution(event_to_schedule):
                    event_to_schedule = hc.possible_solution(event_to_schedule)
                    rule_modifier = True
                    ts_manager.change_start_period(event_to_schedule.date())
            if not ts_manager.has_next() and len(slots_score.keys()) == 0:
                ts_manager.change_start_period(slot.get_end())

        ids_sorted_by_constraints = sorted(
            slots_score, key=slots_score.get, reverse=True)[
            :k]  # Sort and get the top k slots
        # print("IDs sorted by constraints:")
        # for r in ids_sorted_by_constraints:
        #     print(slots[r])

        result = []
        if k > 1:
            for r in ids_sorted_by_constraints:
                result.append(slots[r])
        else:
            result = slots[ids_sorted_by_constraints[0]]

        return result, duration

    def select_slot(self, k=1, alpha=1, beta=1):
        result = []
        for i, event in enumerate(self.events):
            best_slot, duration = self.best_slots(event=event, k=1,
                                                  alpha=alpha, beta=beta)
            event.start_time = best_slot["start"]
            event.end_time = best_slot["end"]
            event.duration = duration
            # event.location =
            # TODO change task status
            result.append(event)
            self._send_invitations(i, event)

        return result
        
    def _send_invitations(self, i, event):
        if self.tasks is not None:
            for attendee in event.participants:
                if attendee.user_id is not None:
                    e_id = None
                    if attendee.user_id == self.tasks[i].initiator_id:
                        e_id = self.tasks[i].event_id
                    new_invitation = Invitation(task=self.tasks[i],
                                                attendee=attendee.user_id,
                                                event_id=e_id,
                                                answered=False,
                                                decision=False)
                    new_invitation.save()

            client = CalendarDBClient()
            client.update_event(self.tasks[i].initiator_id,
                                self.tasks[i].event_id,
                                {"start": event.start_time.isoformat(),
                                 "end": event.end_time.isoformat(),
                                 "duration": event.duration})

    def cleanup(self):
        for event in self.events:
            for attendee in event.participants:
                attendee.cleanup()


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
