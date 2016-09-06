# -*- coding: utf-8 -*-
# Generated by Django 1.10 on 2016-08-31 00:16
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Invitation',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('attendee', models.CharField(max_length=50)),
                ('event_id', models.UUIDField(null=True)),
                ('answered', models.BooleanField(default=False)),
                ('decision', models.BooleanField()),
            ],
        ),
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
        migrations.CreateModel(
            name='Training',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user_id', models.CharField(max_length=50)),
                ('event_type', models.CharField(max_length=30)),
                ('start', models.DateTimeField()),
                ('end', models.DateTimeField()),
                ('duration', models.IntegerField()),
                ('location', models.CharField(default=b'X', max_length=50)),
                ('feedback', models.BooleanField(default=False)),
            ],
        ),
        migrations.AddField(
            model_name='invitation',
            name='task',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='api.SchedulingTask'),
        ),
    ]