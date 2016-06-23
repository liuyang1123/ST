# -*- coding: utf-8 -*-
# Generated by Django 1.9.7 on 2016-06-17 23:38
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='SchedulingTask',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('task_type', models.CharField(choices=[(b'schedule', b'schedule'), (b'reschedule', b'reschedule')], max_length=20)),
                ('status', models.CharField(choices=[(b'pending', b'pending'), (b'started', b'started'), (b'finished', b'finished'), (b'failed', b'failed')], max_length=20)),
                ('event_id', models.UUIDField(editable=False)),
                ('initiator_id', models.IntegerField(editable=False)),
                ('result', models.CharField(choices=[(b'needs_action', b'needs_action'), (b'tentative', b'tentative'), (b'accepted', b'accepted'), (b'declined', b'declined')], max_length=20)),
                ('start_time', models.DateTimeField(null=True)),
                ('tentative_time', models.DateTimeField(null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
        ),
    ]