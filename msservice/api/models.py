from django.db import models
from api.event_module.calendar_client import CalendarDBClient
import datetime
from dateutil.parser import parse


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

            # from api.scheduling_module.scheduler import Scheduler
            # s = Scheduler([event])
            # s.select_best_slot()  # TODO
            pass
        # TODO: if task status == 'started' then update start_time value
        # TODO: if schedule status == 'tentative' then update tentative_time
        # value
