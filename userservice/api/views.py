from rest_framework import routers, views, viewsets, mixins, status
from rest_framework.decorators import list_route
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.decorators import list_route
from rest_framework_jwt.settings import api_settings
from api.models import MyUser, Profile, Application, ApplicationUser  # , AppAuthorization
# , AppAuthorizationSerializer
from api.serializers import UserSerializer, ProfileSerializer, ApplicationSerializer, ApplicationUserSerializer
from api.utils import decode_token
from api.permissions import IsAuthenticated, UserViewSetPermissions, ProfileViewSetPermissions
from rest_framework.renderers import JSONRenderer
from rest_framework_jwt.views import ObtainJSONWebToken

class GetTokenView(ObtainJSONWebToken):
    """
    User registration
    """
    permission_classes = (AllowAny,)
    renderer_classes = (JSONRenderer, )


class RegistrationView(views.APIView):
    """
    User registration
    """
    permission_classes = (AllowAny,)
    renderer_classes = (JSONRenderer, )

    def post(self, request, *args):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.create()

            jwt_payload_handler = api_settings.JWT_PAYLOAD_HANDLER
            jwt_encode_handler = api_settings.JWT_ENCODE_HANDLER
            payload = jwt_payload_handler(user)
            token = jwt_encode_handler(payload)

            return Response({"token": token}, status=status.HTTP_201_CREATED)
        else:
            return Response({
                'error': serializer.errors,
                'message': 'User could not be created with the received data.'
            }, status=status.HTTP_400_BAD_REQUEST)


class ApplicationViewSet(viewsets.ModelViewSet):
    serializer_class = ApplicationSerializer
    permission_classes = (IsAuthenticated,)
    lookup_field = 'client_id'

    def get_queryset(self):
        """
        This view should return a list of all the applications
        for the currently authenticated user.
        """
        decoded_token = decode_token(self.request.META)
        if decoded_token is not None:
            return Application.objects.filter(
                owner=MyUser.objects.get(email=decoded_token['email']))
        return []

    def perform_update(self, serializer):
        decoded_token = decode_token(self.request.META)
        if decoded_token is not None:
            serializer.save(
                owner=MyUser.objects.get(
                    email=decoded_token['email']))

    def perform_create(self, serializer):
        decoded_token = decode_token(self.request.META)
        if decoded_token is not None:
            serializer.save(
                owner=MyUser.objects.get(
                    email=decoded_token['email']))


class AllUsersView(views.APIView):

    permission_classes = (AllowAny,)

    def get(self, request):
        serializer = UserSerializer(MyUser.objects.all(), many=True)

        return Response(serializer.data, status=status.HTTP_200_OK)


class UserView(views.APIView):

    permission_classes = (IsAuthenticated,)

    def get(self, request):
        decoded_token = decode_token(request.META)
        serializer = UserSerializer(
            MyUser.objects.get(
                email=decoded_token['email']))

        return Response(serializer.data, status=status.HTTP_200_OK)

    def put(self, request):
        # Si cambia el email puede generar problemas serios, por eso se retorna
        # un nuevo token
        decoded_token = decode_token(request.META)
        serializer = UserSerializer(
            MyUser.objects.get(email=decoded_token['email']),
            data=request.data,
            partial=True)

        if serializer.is_valid():
            user = serializer.save()

            jwt_payload_handler = api_settings.JWT_PAYLOAD_HANDLER
            jwt_encode_handler = api_settings.JWT_ENCODE_HANDLER
            payload = jwt_payload_handler(user)
            token = jwt_encode_handler(payload)

            return Response({"token": token}, status=status.HTTP_200_OK)
        else:
            return Response({
                'data': serializer.errors,
                'message': 'User could not be created with the received data.'
            }, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request):
        decoded_token = decode_token(request.META)
        MyUser.objects.get(email=decoded_token['email']).delete()

        return Response("User deleted!", status=status.HTTP_204_NO_CONTENT)


class UserViewSet(viewsets.ModelViewSet):
    serializer_class = UserSerializer
    permission_classes = (UserViewSetPermissions,)
    lookup_field = 'profile__profile_id'

    def get_queryset(self):
        client_id = self.kwargs['client_id']
        users = ApplicationUser.objects.filter(
            application__client_id=client_id).values_list(
            'user', flat=True)
        return MyUser.objects.filter(pk__in=users)

    def filter_queryset(self, queryset):
        decoded_token = decode_token(self.request.META)
        try:
            # If the user is the owner
            app = Application.objects.get(client_id=self.kwargs['client_id'],
                                          owner__email=decoded_token['email'])
            return queryset
        except Application.DoesNotExist:
            return queryset.filter(email=decoded_token['email'])
        return queryset

    def create(self, request, *args, **kwargs):
        app = None
        try:
            app = Application.objects.get(client_id=self.kwargs['client_id'])
        except Application.DoesNotExist:
            return Response({
                'message': 'Invalid client_id.'
            }, status=status.HTTP_400_BAD_REQUEST)

        serializer = UserSerializer(data=request.data)

        if serializer.is_valid():
            user = serializer.create()

            application_user = ApplicationUser(application=app,
                                               user=user)
            application_user.save()

            jwt_payload_handler = api_settings.JWT_PAYLOAD_HANDLER
            jwt_encode_handler = api_settings.JWT_ENCODE_HANDLER
            payload = jwt_payload_handler(user)
            token = jwt_encode_handler(payload)

            return Response({"token": token}, status=status.HTTP_201_CREATED)
        else:
            return Response({
                'data': serializer.errors,
                'message': 'User could not be created with the received data.'
            }, status=status.HTTP_400_BAD_REQUEST)

    @list_route(methods=['post'])
    def get_user_by_email(self, request, *args, **kwargs):
        email = request.data['email']

        app_user = ApplicationUser.objects.filter(
            application__client_id=self.kwargs['client_id'], user__email=email)

        if app_user.exists():
            return Response(
                {"user_id": app_user[0].user.id}, status=status.HTTP_200_OK)
        else:
            return Response({'message': 'User does not exists.'},
                            status=status.HTTP_400_BAD_REQUEST)


class ProfileViewSet(viewsets.ModelViewSet):

    serializer_class = ProfileSerializer
    permission_classes = (
        ProfileViewSetPermissions,
    )  # TODO Edit: OwnerPermission
    lookup_field = 'profile_id'

    def get_queryset(self):
        client_id = self.kwargs['client_id']
        users = ApplicationUser.objects.filter(
            application__client_id=client_id).values_list(
            'user', flat=True)
        return Profile.objects.filter(user__in=users)

    def filter_queryset(self, queryset):
        decoded_token = decode_token(self.request.META)
        try:
            # If the user is the owner
            app = Application.objects.get(client_id=self.kwargs['client_id'],
                                          owner__email=decoded_token['email'])
            return queryset
        except Application.DoesNotExist:
            return queryset.filter(user__email=decoded_token['email'])
        return queryset

    def create(self, request, *args, **kwargs):
        app = None
        try:
            app = Application.objects.get(client_id=self.kwargs['client_id'])
        except Application.DoesNotExist:
            return Response({
                'message': 'Invalid client_id.'
            }, status=status.HTTP_400_BAD_REQUEST)

        decoded_token = decode_token(request.META)
        instance = Profile.objects.get(
            user=MyUser.objects.get(
                email=decoded_token['email']))
        partial = kwargs.pop('partial', False)
        serializer = self.get_serializer(
            instance, data=request.data, partial=partial)
        if serializer.is_valid():
            self.perform_update(serializer)
            return Response(serializer.data, status=status.HTTP_200_OK)

        return Response({
            'data': serializer.errors,
            'message': 'Profile could not be updated with the received data.'
        }, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, *args):
        pass

# class AppAuthorizationViewSet(viewsets.ModelViewSet):
#     queryset = AppAuthorization.objects.all()
#     serializer_class = AppAuthorizationSerializer
#     permission_classes = (IsAuthenticated,) # TODO Edit: OwnerPermission


# Routers provide an easy way of automatically determining the URL conf.
router = routers.DefaultRouter()
router.register(r'(?P<client_id>\S+)/users', UserViewSet, base_name='users')
router.register(
    r'(?P<client_id>\S+)/profiles',
    ProfileViewSet,
    base_name='profiles')
router.register(r'applications', ApplicationViewSet, base_name='applications')
# router.register(r'apps', AppAuthorizationViewSet)
