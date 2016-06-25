from dateutil.parser import parse
from datetime import date, timedelta
from django.db import models
from dateutil.parser import parse
from api.event_module.calendar_client import CalendarDBClient
from api.event_module.event import Event
from api.scheduling_module.scheduler import Scheduler
from api.learning_module.profile.profile import Attendee
from api.event_module.manager import EVENT_TYPES_DICT, DEFAULT_EVENT_TYPE
from api.event_module.event_type import EventType

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

    def save(self, *args, **kwargs):
        super(SchedulingTask, self).save(*args, **kwargs)
        if self.status == 'pending':
            # from .negotiation import schedule
            # event = get_event_by_id(self.event_id)
            # schedule(scheduling_task_id=self.id, scheduling_task=self)

            # 1. Get the event object with the self.event_id attribute
            client = CalendarDBClient()
            event_json = client.get_event(self.event_id)

            if event_json is not None and len(event_json) > 0:
                event_json = event_json[0]

                # Get the valid range to schedule the event
                start = event_json.get('start', None) # Si es None -> Tomorrow
                if start is not None and start!='':
                    start = parse(start)
                else:
                    start = date.today() + timedelta(days=1)
                end = event_json.get('end', None) # Si es None +15 days
                if end is not None and end!='':
                    end = parse(end)
                else:
                    end = date.today() + timedelta(days=15)

                # Get the profile of every participant
                participants = list()
                for email in event_json.get('attendees', []):
                    attendee = Attendee(email=email.replace(" ", ""))
                    participants.append(attendee)
                participants.append(Attendee(user_id=event_json.get('user_id')))

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
                              location=event_json.get('location', ''))

                # 2. Create a scheduler object
                s = Scheduler([event])
                # 3. Select the best (n) timeslots
                s.select_slot()
                # Close the connection
                s.cleanup()

            # 4. Create the new updates mechanisms (started, tentative, confirm)
            # 5. Send users invitations

        # TODO: if task status == 'started' then update start_time value
        # TODO: if schedule status == 'tentative' then update tentative_time
        # value
