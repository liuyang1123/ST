import json
from rest_framework import views, viewsets, status, routers
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework.decorators import list_route, detail_route
from api.models import SchedulingTask, Invitation
from api.permissions import IsAuthenticated
from api.serializers import SchedulingTaskSerializer, EventSerializer, PreferenceSerializer, InvitationSerializer
from api.utils import decode_token, get_token
from api.event_module.calendar_client import CalendarDBClient
from api.learning_module.hard_constraints.preferences.manager import UserPreferencesManager

from api.learning_module.soft_constraints.data_utils import *
from api.learning_module.soft_constraints.bayesian_network import BayesianNetworkModel

# class SchedulingTasksViewSet(viewsets.ModelViewSet):
#     queryset = SchedulingTask.objects.all()
#     serializer_class = SchedulingTaskSerializer
#     # IsStaff TODO Update the UserService to know if the authenticated user is
#     # a Staff Member -> This way users can know the status of their event
#     # scheduling, and staff member can track general statuses
#     permission_classes = (IsAuthenticated,)
#
#     def perform_update(self, serializer):
#         # Actualizar event_id, etc.
#         pass
#
#     def perform_create(self, serializer):
#         pass


class ScheduleViewSet(viewsets.ViewSet):
    permission_classes = (IsAuthenticated,)

    def list(self, request):
        decoded_token = decode_token(request.META)
        serializer = SchedulingTaskSerializer(SchedulingTask.objects.filter(
            initiator_id=decoded_token['user_id']), many=True)

        return Response(serializer.data, status=status.HTTP_200_OK)

    def create(self, request):
        decoded_token = decode_token(request.META)
        token = get_token(request.META)
        serializer = EventSerializer(data=request.data)

        if serializer.is_valid():
            client = CalendarDBClient()

            event_id = client.add_event(token, serializer.data)

            st = SchedulingTask(
                task_type='schedule',
                status='pending',
                event_id=event_id,
                start_time=None,
                tentative_time=None,
                initiator_id=decoded_token['user_id'],
                result='needs_action'
            )
            st.save()

            return Response({"message": "Event created."},
                            status=status.HTTP_201_CREATED)
        return Response({
            'data': serializer.errors,
            'message': 'Event could not be created with the given data.'
        }, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, pk=None):
        if pk is None:
            return Response({
                'message': 'Provide the object id.'
            }, status=status.HTTP_400_BAD_REQUEST)

        decoded_token = decode_token(request.META)

        SchedulingTask.objects.filter(id=pk).delete()

        return Response({"message": "Deleted."}, status=status.HTTP_200_OK)

class InvitationViewSet(viewsets.ViewSet):
    permission_classes = (IsAuthenticated,)

    def list(self, request):
        decoded_token = decode_token(request.META)
        serializer = InvitationSerializer(Invitation.objects.filter(
            attendee=decoded_token['user_id']), many=True)

        return Response(serializer.data, status=status.HTTP_200_OK)

    @list_route(methods=['get'])
    def pending(self, request):
        decoded_token = decode_token(request.META)
        serializer = InvitationSerializer(Invitation.objects.filter(
            attendee=decoded_token['user_id'],
            answered=False), many=True)

        return Response(serializer.data, status=status.HTTP_200_OK)

    @detail_route(methods=['post'])
    def accept(self, request, pk=None):
        decoded_token = decode_token(request.META)
        q = Invitation.objects.filter(attendee=decoded_token['user_id'],
                                      pk=pk)
        if q.exists():
            q = q[0]
            q.answered = True
            q.decision = True
            q.save()

        return Response({"data": "Invitation accepted."},
                        status=status.HTTP_200_OK)

    @detail_route(methods=['post'])
    def decline(self, request, pk=None):
        decoded_token = decode_token(request.META)
        q = Invitation.objects.filter(attendee=decoded_token['user_id'],
                                      pk=pk)
        if q.exists():
            q = q[0]
            q.answered = True
            q.decision = False
            q.save()

        return Response({"data": "Invitation declined."},
                        status=status.HTTP_200_OK)


class BNTrainingView(viewsets.ViewSet):
    permission_classes = (IsAuthenticated,)

    def create(self, request):
        """
        Triggers the learning mechanism.
        """
        decoded_token = decode_token(request.META)
        user_id = decoded_token['user_id']

        # TODO Do this as a Celery task
        datasets_bn = read_data_sets_bn(user_id)
        bn = BayesianNetworkModel(user_id)
        bn.build_model()
        data, labels = datasets_bn.train.next_batch()
        bn.train(data, labels)
        bn.save()
        bn.close()
        return Response({"message": "Training started."},
                        status=status.HTTP_200_OK)

    @list_route(methods=['post'])
    def infer(self, request):
        """
        Triggers the learning mechanism.
        """
        decoded_token = decode_token(request.META)
        user_id = decoded_token['user_id']

        bn = BayesianNetworkModel(user_id)
        obs = {}
        try:
            obs = json.loads(request.data['observation'])
        except:
            pass
        result = bn.predict(obs)
        bn.close()

        return Response(json.dumps(result), status=status.HTTP_200_OK)


class PreferenceViewSet(viewsets.ViewSet):
    permission_classes = (IsAuthenticated,)

    def create(self, request):
        decoded_token = decode_token(request.META)

        serializer = PreferenceSerializer(data=request.data)
        if serializer.is_valid():
            preference_manager = UserPreferencesManager()

            data = serializer.data
            data['user_id'] = decoded_token['user_id']
            data['attributes'] = json.loads(data['attributes'])

            inserted = preference_manager.insert(data)
            preference_manager.close()

            return Response({"message": "Preference created.",
                             "data": inserted}, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def list(self, request):
        decoded_token = decode_token(request.META)

        preference_manager = UserPreferencesManager()
        serializer = PreferenceSerializer(
            preference_manager.list(decoded_token['user_id']),
            many=True)
        preference_manager.close()

        return Response(serializer.data, status=status.HTTP_200_OK)

    # def retrieve(self, request, pk=None):
    #     if pk is None:
    #         return Response({
    #             'message': 'Provide the object id.'
    #         }, status=status.HTTP_400_BAD_REQUEST)
    #
    #     decoded_token = decode_token(request.META)
    #     return Response(self.get_preference(
    #         decoded_token['user_id'], pk), status=status.HTTP_200_OK)

    def update(self, request, pk=None):
        if pk is None:
            return Response({
                'message': 'Provide the object id.'
            }, status=status.HTTP_400_BAD_REQUEST)

        decoded_token = decode_token(request.META)

        preference_manager = UserPreferencesManager()
        # TODO Catch if not exists
        pref_instance = preference_manager.get(decoded_token['user_id'], pk)

        if len(pref_instance) == 0:
            preference_manager.close()
            return Response({
                'message': 'Provide the object id.'
            }, status=status.HTTP_400_BAD_REQUEST)
        else:
            pref_instance = pref_instance[0]

        serializer = PreferenceSerializer(
            instance=pref_instance,
            data=request.data,
            partial=True)

        if serializer.is_valid():
            data = serializer.data
            data['user_id'] = decoded_token['user_id']
            data['preference_name'] = serializer.validated_data.get(
                'preference_name', data['preference_name'])
            data['preference_type'] = serializer.validated_data.get(
                'preference_type', data['preference_type'])
            data['event_type'] = serializer.validated_data.get(
                'event_type', data['event_type'])
            data['attributes'] = json.loads(serializer.validated_data.get(
                'attributes', data['attributes']))

            updated = preference_manager.update(pk, data)
            preference_manager.close()

            return Response({"data": updated}, status=status.HTTP_200_OK)

        preference_manager.close()

        return Response(serializer.errors)

    def destroy(self, request, pk=None):
        if pk is None:
            return Response({
                'message': 'Provide the object id.'
            }, status=status.HTTP_400_BAD_REQUEST)

        decoded_token = decode_token(request.META)

        preference_manager = UserPreferencesManager()
        deleted = preference_manager.delete(decoded_token['user_id'], pk)
        preference_manager.close()

        return Response({"data": deleted}, status=status.HTTP_200_OK)

    @list_route(methods=['post'])
    def default(self, request):
        decoded_token = decode_token(request.META)

        preference_manager = UserPreferencesManager()
        preference_manager.delete_all(decoded_token['user_id'])
        # for elem in DEFAULT_PREFERENCES:
        #     data = elem
        #     data['user_id'] = decoded_token['user_id']
        #     preference_manager.insert(data)
        preference_manager.close()

        return Response({"data": "All the preferences has been created"},
                        status=status.HTTP_201_CREATED)


# Routers provide an easy way of automatically determining the URL conf.
router = routers.DefaultRouter()
router.register(r'scheduling', ScheduleViewSet, base_name='scheduling')
router.register(r'invitation', InvitationViewSet, base_name='invitation')
router.register(r'bayesiannetwork', BNTrainingView,
                base_name='bayesiannetwork')
router.register(r'preferences', PreferenceViewSet, base_name='preferences')
# ModelViewSet:
# router.register(r'scheduling-feed', SchedulingTasksViewSet,
#                 base_name='scheduling_feed')
