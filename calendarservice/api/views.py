import requests
import json
from rest_framework import routers, viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import list_route
from api.models import Calendar, Event
from api.serializers import CalendarSerializer, EventSerializer
from api.utils import decode_token
from api.permissions import IsAuthenticated
from rest_framework.permissions import AllowAny
from django.utils.timezone import now, timedelta
from rest_framework.renderers import JSONRenderer


# TODO Set a default Calendar so that there the calendar_id field is optional

class CalendarViewSet(viewsets.ViewSet):
    permission_classes = (IsAuthenticated,)
    renderer_classes = (JSONRenderer, )

    def list(self, request):
        decoded_token = decode_token(request.META)

        calendar = Calendar()
        serializer = CalendarSerializer(
            calendar.list(decoded_token['user_id']),
            many=True)
        calendar.close()

        return Response(serializer.data, status=status.HTTP_200_OK)

    def create(self, request):
        decoded_token = decode_token(request.META)
        serializer = CalendarSerializer(data=request.data)
        if serializer.is_valid():
            calendar = Calendar()
            data = serializer.data
            data['user_id'] = decoded_token['user_id']
            inserted = calendar.insert(data)
            calendar.close()
            return Response({"message": "Calendar created.",
                             "data": inserted}, status=status.HTTP_201_CREATED)
        return Response({
            'data': serializer.errors,
            'message': 'Calender could not be created with the given data.'
        }, status=status.HTTP_400_BAD_REQUEST)

    def retrieve(self, request, pk=None):
        if pk is None:
            return Response({
                'message': 'Provide the object id.'
            }, status=status.HTTP_400_BAD_REQUEST)

        decoded_token = decode_token(request.META)

        calendar = Calendar()
        serializer = CalendarSerializer(
            calendar.get(decoded_token['user_id'], pk),
            many=True)
        calendar.close()

        return Response(serializer.data, status=status.HTTP_200_OK)

    def update(self, request, pk=None):
        if pk is None:
            return Response({
                'message': 'Provide the object id.'
            }, status=status.HTTP_400_BAD_REQUEST)

        decoded_token = decode_token(request.META)

        calendar = Calendar()
        updated = calendar.update(decoded_token['user_id'], pk, request.data)
        calendar.close()

        return Response({"data": updated}, status=status.HTTP_200_OK)

    def partial_update(self, request, pk=None):
        return self.update(request, pk)

    def destroy(self, request, pk=None):
        if pk is None:
            return Response({
                'message': 'Provide the object id.'
            }, status=status.HTTP_400_BAD_REQUEST)

        decoded_token = decode_token(request.META)

        calendar = Calendar()
        deleted = calendar.delete(decoded_token['user_id'], pk)
        calendar.close()

        return Response({"data": deleted}, status=status.HTTP_200_OK)


class EventsViewSet(viewsets.ViewSet):
    permission_classes = (IsAuthenticated,)
    renderer_classes = (JSONRenderer, )

    def list(self, request, calendar_id):
        decoded_token = decode_token(request.META)

        event = Event()
        serializer = EventSerializer(
            event.list_events(decoded_token['user_id'], calendar_id),
            many=True)
        event.close()

        return Response(serializer.data, status=status.HTTP_200_OK)

    def create(self, request, calendar_id):
        # self.kwargs['calendar_id']
        decoded_token = decode_token(request.META)

        calendar = Calendar()
        obj = calendar.get(decoded_token['user_id'], calendar_id)
        calendar.close()

        if len(obj) < 1:
            calendar_id = -1
            # return Response({"message": "calendar_id does not exists."},
            #                status=status.HTTP_404_NOT_FOUND)

        serializer = EventSerializer(data=request.data)
        if serializer.is_valid():
            event = Event()

            data = serializer.data
            data['calendar_id'] = calendar_id
            data['created'] = now()
            data['updated'] = now()
            data['user_id'] = decoded_token['user_id']

            inserted = event.insert_event(data)
            event.close()

            return Response({"message": "Event created.",
                             "data": inserted}, status=status.HTTP_201_CREATED)

        print("OK")
        print(serializer.errors)

        return Response(serializer.errors)

    def retrieve(self, request, calendar_id, pk=None):
        if pk is None:
            return Response({
                'message': 'Provide the object id.'
            }, status=status.HTTP_400_BAD_REQUEST)

        decoded_token = decode_token(request.META)

        event = Event()
        serializer = EventSerializer(
            event.get_event(decoded_token['user_id'], calendar_id, pk),
            many=True)
        event.close()

        return Response(serializer.data, status=status.HTTP_200_OK)

    def update(self, request, calendar_id, pk=None):
        if pk is None:
            return Response({
                'message': 'Provide the object id.'
            }, status=status.HTTP_400_BAD_REQUEST)

        decoded_token = decode_token(request.META)

        calendar = Calendar()
        obj = calendar.get(decoded_token['user_id'], calendar_id)
        calendar.close()

        if len(obj) < 1:
            return Response({"message": "calendar_id does not exists."},
                            status=status.HTTP_404_NOT_FOUND)

        event = Event()
        # TODO Revisarlo con el serializer
        data = request.data
        data['updated'] = now()
        updated = event.update_event(calendar_id, pk, data)
        event.close()

        return Response({"data": updated}, status=status.HTTP_200_OK)

    def partial_update(self, request, calendar_id, pk=None):
        return self.update(self, request, pk)

    def destroy(self, request, calendar_id, pk=None):
        if pk is None:
            return Response({
                'message': 'Provide the object id.'
            }, status=status.HTTP_400_BAD_REQUEST)

        decoded_token = decode_token(request.META)

        calendar = Calendar()
        obj = calendar.get(decoded_token['user_id'], calendar_id)
        calendar.close()

        if len(obj) < 1:
            return Response({"message": "calendar_id does not exists."},
                            status=status.HTTP_404_NOT_FOUND)

        event = Event()
        deleted = event.delete_event(calendar_id, pk)
        event.close()

        return Response({"data": deleted}, status=status.HTTP_200_OK)

    @list_route(methods=['post'])
    def free_busy(self, request, calendar_id):
        decoded_token = decode_token(request.META)

        from_date = request.data('from_date', now() - timedelta(days=14))
        to_date = request.data('to_date', now() + timedelta(days=14))

        event = Event()
        serializer = EventSerializer(
            event.free_busy(decoded_token['user_id'],
                            from_date, to_date),
            many=True)
        event.close()

        return Response(serializer.data, status=status.HTTP_200_OK)

    @list_route(methods=['post'])
    def available(self, request, calendar_id):
        decoded_token = decode_token(request.META)

        from_date = request.data.get('from_date', None)
        to_date = request.data.get('to_date', None)

        if from_date is None or to_date is None:
            return Response({
                'message': 'Invalid information provided, please include values for from_date and to_date.'
            }, status=status.HTTP_400_BAD_REQUEST)

        event = Event()
        result = event.available(decoded_token['user_id'],
                                 from_date, to_date)
        event.close()

        return Response({"available": result}, status=status.HTTP_200_OK)


class QViewSet(viewsets.ViewSet):
    permission_classes = (AllowAny,)
    renderer_classes = (JSONRenderer, )

    def is_valid(self, client_id, user_id):
        # TODO
        return True

    def update(self, request, client_id, user_id, pk=None):
        if pk is None:
            return Response({
                'message': 'Provide the object id.'
            }, status=status.HTTP_400_BAD_REQUEST)

        event = Event()
        # TODO revisarlo con el serializer
        data = request.data.copy()
        data['updated'] = now()
        updated = event.update_event_with_pk(pk, data)
        event.close()

        return Response({"data": updated}, status=status.HTTP_200_OK)

    def retrieve(self, request, client_id, user_id, pk=None):
        if pk is None:
            return Response({
                'message': 'Provide the object id.'
            }, status=status.HTTP_400_BAD_REQUEST)

        # TODO Add a different endpoint without using user_id
        if not self.is_valid(client_id,
                             user_id) or not user_id == "554-586-725":
            return Response({"message": "Not found."},
                            status=status.HTTP_404_NOT_FOUND)

        event = Event()
        serializer = EventSerializer(
            event.get_event_with_pk(pk),
            many=True)
        event.close()

        return Response(serializer.data, status=status.HTTP_200_OK)

    @list_route(methods=['get'])
    def events(self, request, client_id, user_id):
        if not self.is_valid(client_id, user_id):
            return Response({"message": "Not found."},
                            status=status.HTTP_404_NOT_FOUND)

        event = Event()
        serializer = EventSerializer(
            event.list_all_events_for_training(int(user_id)), many=True)
        event.close()

        return Response(serializer.data, status=status.HTTP_200_OK)

    @list_route(methods=['post'])
    def free_busy(self, request, client_id, user_id):
        if not self.is_valid(client_id, user_id):
            return Response({"message": "Not found."},
                            status=status.HTTP_404_NOT_FOUND)

        from_date = request.data.get('from_date',
                                     (now() - timedelta(days=14)).isoformat())
        to_date = request.data.get('to_date',
                                   (now() + timedelta(days=14)).isoformat())

        event = Event()
        serializer = EventSerializer(
            event.free_busy(int(user_id), from_date, to_date), many=True)
        event.close()

        return Response(serializer.data, status=status.HTTP_200_OK)

    @list_route(methods=['post'])
    def qavailable(self, request, client_id, user_id):
        if not self.is_valid(client_id, user_id):
            return Response({"message": "Not found."},
                            status=status.HTTP_404_NOT_FOUND)

        from_date = request.data.get('from_date',
                                     (now() - timedelta(hours=3)).isoformat())
        to_date = request.data.get('to_date',
                                   (now() - timedelta(hours=3)).isoformat())

        if from_date is None or to_date is None:
            return Response({
                'message': 'Invalid information provided, please include values for from_date and to_date.'
            }, status=status.HTTP_400_BAD_REQUEST)

        event = Event()
        result = event.available(int(user_id), from_date, to_date)
        event.close()

        return Response({"available": result}, status=status.HTTP_200_OK)


class QEventViewSet(viewsets.ViewSet):
    permission_classes = (IsAuthenticated,)
    renderer_classes = (JSONRenderer,)

    @list_route(methods=['post'])
    def free_busy(self, request):
        decoded_token = decode_token(request.META)

        from_date = request.data.get('from_date',
                                     (now() - timedelta(days=14)).isoformat())
        to_date = request.data.get('to_date',
                                   (now() + timedelta(days=14)).isoformat())

        event = Event()
        serializer = EventSerializer(
            event.free_busy(int(decoded_token['user_id']), from_date, to_date),
            many=True)
        event.close()

        return Response(serializer.data, status=status.HTTP_200_OK)

    def destroy(self, request, pk=None):
        decoded_token = decode_token(request.META)

        event = Event()
        deleted = event.delete_event_with_user_id(
            pk, int(decoded_token['user_id']))
        event.close()

        return Response({"data": deleted}, status=status.HTTP_200_OK)

    def create(self, request):
        decoded_token = decode_token(request.META)

        serializer = EventSerializer(data=request.data)
        if serializer.is_valid():
            event = Event()

            data = serializer.data
            data['calendar_id'] = -1
            data['created'] = now()
            data['updated'] = now()
            data['user_id'] = decoded_token['user_id']

            inserted = event.insert_event(data)

            result = {}
            if inserted.get('inserted', 0) == 1:
                e = EventSerializer(
                    event.get_event_with_pk(inserted['generated_keys'][0]),
                    many=True)
                if len(e.data) > 0:
                    result = e.data[0]

            event.close()

            return Response(result, status=status.HTTP_201_CREATED)

        return Response(serializer.errors)


# Routers provide an easy way of automatically determining the URL conf.
router = routers.DefaultRouter()
router.register(
    r'c/(?P<client_id>\S+)/u/(?P<user_id>\S+)/events',
    QViewSet,
    base_name='eclient')
router.register(r'calendars', CalendarViewSet, base_name='calendars')
router.register(
    r'(?P<calendar_id>\S+)/events',
    EventsViewSet,
    base_name='events')
router.register(
    r'myevents',
    QEventViewSet,
    base_name='qevents')
