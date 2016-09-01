from dateutil.parser import parse
from datetime import date, timedelta
from django.db import models
from dateutil.parser import parse
from api.event_module.calendar_client import CalendarDBClient
from api.event_module.event import Event
from api.learning_module.profile.profile import Attendee
from api.event_module.manager import EVENT_TYPES_DICT, DEFAULT_EVENT_TYPE
from api.event_module.event_type import EventType

# TODO Improve the usage of task / schedule statuses.
# TODO Use a tentative event model instead of calling the CS?


class SchedulingTask(models.Model):
    TASK_TYPES = (
        ('schedule', 'schedule'),
        ('reschedule', 'reschedule'),
    )

    TASK_STATUSES = (
        ('pending', 'pending'),
        ('started', 'started'),
        ('finished', 'finished'),
        ('failed', 'failed'),
    )

    SCHEDULE_STATUSES = (
        ('needs_action', 'needs_action'),
        ('tentative', 'tentative'),
        ('accepted', 'accepted'),
        ('declined', 'declined'),
    )

    task_type = models.CharField(choices=TASK_TYPES, max_length=20)
    status = models.CharField(choices=TASK_STATUSES, max_length=20)
    event_id = models.UUIDField(editable=False)
    # initiator_id = models.UUIDField(editable=False)
    initiator_id = models.IntegerField(editable=False)
    result = models.CharField(choices=SCHEDULE_STATUSES, max_length=20)

    start_time = models.DateTimeField(null=True)
    tentative_time = models.DateTimeField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def get_event(self):
        client = CalendarDBClient()
        return client.get_event(self.event_id)

    def save(self, *args, **kwargs):
        super(SchedulingTask, self).save(*args, **kwargs)
        from api.scheduling_module.scheduler import Scheduler
        if self.status == 'pending':
            # 1. Get the event object with the self.event_id attribute
            event_json = self.get_event()

            if event_json is not None and len(event_json) > 0:
                event_json = event_json
                event_json["created"] = parse(event_json["created"])
                event_json["updated"] = parse(event_json["updated"])
                
                # Get the valid range to schedule the event
                start = event_json.get('start', None)  # Si es None -> Tomorrow
                if start is not None and start != '':
                    start = parse(start)
                    try:
                        start = start.date()
                    except:
                        pass
                else:
                    start = date.today() + timedelta(days=1)
                end = event_json.get('end', None)  # Si es None +15 days
                if end is not None and end != '':
                    end = parse(end)
                    try:
                        end = end.date()
                    except:
                        pass
                else:
                    end = date.today() + timedelta(days=7)

                print("Scheduler")

                # Get the profile of every participant
                participants = list()
                for email in event_json.get('attendees', []):
                    attendee = Attendee(email=email.replace(" ", ""))
                    participants.append(attendee)
                participants.append(
                    Attendee(
                        user_id=event_json.get('user_id')))

                e_type = EVENT_TYPES_DICT.get(
                    str(event_json.get("categories",
                                       DEFAULT_EVENT_TYPE)).lower())

                e_type_obj = EventType(event_json.get("event_type",
                                                      DEFAULT_EVENT_TYPE),
                                       e_type["is_live"],
                                       e_type["unique_per_day"])

                # TODO Figure out a good way to gather locations knowledge
                event = Event(participants=participants,
                              event_type=e_type_obj,
                              description=event_json.get('description', ''),
                              duration=int(event_json.get('duration', 0)),
                              start_time=start,
                              end_time=end,
                              location=event_json.get('location', ''),
                              attr = event_json)

                # 2. Create a scheduler object
                s = Scheduler([event], [self])
                # 3. Select the best (n) timeslots
                re = s.select_slot()
                # Close the connection
                s.cleanup()

                return re

            # 4. Create the new updates mechanisms (started, tentative, confirm)
            # 5. Send users invitations

        # TODO: if task status == 'started' then update start_time value
        # TODO: if schedule status == 'tentative' then update tentative_time
        # value


class Training(models.Model):
    user_id = models.CharField(max_length=50)
    event_type = models.CharField(max_length=30)
    start = models.DateTimeField()
    end = models.DateTimeField()
    duration = models.IntegerField()
    location = models.CharField(max_length=50, default='X')
    feedback = models.BooleanField(default=False)
    # participants
    # location


class Invitation(models.Model):
    task = models.ForeignKey(SchedulingTask)
    attendee = models.CharField(max_length=50)
    event_id = models.UUIDField(null=True)
    answered = models.BooleanField(default=False)
    decision = models.BooleanField()

    def save(self, *args, **kwargs):
        super(Invitation, self).save(*args, **kwargs)
        if self.answered:
            event = self.task.get_event()

            if not self.decision:
                # Save the event as negative sampling
                t = Training(user_id=self.attendee,
                             event_type=event.get("categories"),
                             start=parse(event.get("start")),
                             end=parse(event.get("end")),
                             duration=int(event.get("duration")),
                             feedback=False)
                t.save()
            else:
                invitations = Invitation.objects.filter(task=self.task)
                ans = True
                dec = True
                for inv in invitations:
                    if not inv.answered:
                        ans = False
                        break
                    else:
                        dec = dec & inv.decision
                if ans:
                    if dec:
                        # TODO Send confirmations
                        # TODO Update users calendars. Update event_id
                        # TODO The initiator may already have an event assigned
                        # TODO Change task status
                        # TODO Remove this training example
                        pass
                    else:
                        # TODO Initiate a rescheduling process, call the
                        # best_slots with the invalid parameter
                        pass
                # TODO Figure out if it would be a good idea to create training
                # data from this.
                t = Training(user_id=self.attendee,
                             event_type=event.get("categories"),
                             start=parse(event.get("start")),
                             end=parse(event.get("end")),
                             duration=int(event.get("duration")),
                             feedback=True)
                t.save()
