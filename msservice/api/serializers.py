from rest_framework import serializers
from api.models import SchedulingTask, Invitation
# from api.hc.Preferences.PreferenceManager import PREFERENCE_NAME, EVENT_TYPE


class SchedulingTaskSerializer(serializers.ModelSerializer):

    class Meta:
        model = SchedulingTask


class InvitationSerializer(serializers.ModelSerializer):

    class Meta:
        model = Invitation


class EventSerializer(serializers.Serializer):
    # autogenerated by rethinkdb
    id = serializers.CharField(max_length=36, required=False)
    calendar_id = serializers.CharField(
        max_length=36)  # self.kwargs['calendar_id']
    user_id = serializers.CharField(max_length=36, read_only=True)
    summary = serializers.CharField(
        max_length=140, allow_blank=True, default="")
    description = serializers.CharField(
        max_length=200, allow_blank=True, default="")
    deleted = serializers.BooleanField(default=False)
    # Can be blank = To be determined
    start = serializers.CharField(max_length=50, allow_blank=True, default="")
    # Can be blank = To be determined
    end = serializers.CharField(max_length=50, allow_blank=True, default="")
    # start - end or a specific duration when start y end are null
    duration = serializers.IntegerField(allow_null=True, default=-1)
    location = serializers.CharField(
        max_length=140, allow_blank=True, default="")
    participation_status = serializers.CharField(
        max_length=20, default="unknown", allow_blank=True)
    attendees = serializers.ListField(allow_null=True)
    transparency = serializers.CharField(
        max_length=20, default="unknown", allow_blank=True)
    event_status = serializers.CharField(
        max_length=20, default="unknown", allow_blank=True)
    categories = serializers.CharField(
        max_length=20, allow_blank=True, default="meeting")
    is_fixed = serializers.BooleanField(default=False)
    created = serializers.DateTimeField(read_only=True)  # Set on inserted
    updated = serializers.DateTimeField(
        read_only=True)  # Set on inserted, and update


# TODO add validation for event_types and preferences_types
class PreferenceSerializer(serializers.Serializer):
    # autogenerated by rethinkdb
    id = serializers.CharField(max_length=36, read_only=True)
    user_id = serializers.IntegerField(required=False)
    preference_name = serializers.CharField(
        max_length=40, allow_blank=True, default="")
    preference_type = serializers.CharField(
        max_length=40, allow_blank=True, default="")
    event_type = serializers.CharField(
        max_length=40, allow_blank=True, default="")
    # attributes = serializers.JSONField(binary=True)
    attributes = serializers.DictField()

    # def validate_preference_name(self, preference_name):
    #     if preference_name.lower() in PREFERENCE_NAME:
    #         return preference_name
    #     raise serializers.ValidationError("Preference name unknown")

    # def validate_transparency(self, event_type):
    #     if event_type.lower() in EVENT_TYPE:
    #         return event_type
    #     raise serializers.ValidationError("Event type unknown")
