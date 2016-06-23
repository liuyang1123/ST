from rest_framework import serializers
from api.models import MyUser, Profile, Application, ApplicationUser #, AppAuthorization


class ProfileSerializer(serializers.ModelSerializer):

    class Meta:
        model = Profile
        exclude = ('user',)
        extra_kwargs = {'profile_id': {'read_only': True}}


class UserSerializer(serializers.ModelSerializer):
    profile = ProfileSerializer(read_only=True)
    # authentications = AppAuthorizationSerializer(read_only=True, many=True)
    # auth_password = serializers.CharField(style={'input_type': 'password'})

    class Meta:
        model = MyUser
        fields = ('email', 'first_name', 'last_name', 'default_tzid',
                  'id', 'password', 'profile')
        extra_kwargs = {'password': {'write_only': True}}
        depth = 1

    def create(self):
        user = MyUser(email=self.validated_data['email'],
                      first_name=self.validated_data['first_name'],
                      last_name=self.validated_data['last_name'],
                      default_tzid=self.validated_data['default_tzid'])
        if self.validated_data['password']:
            user.set_password(self.validated_data['password'])
        else:
            user.set_password('f1a28a8a-8308-4c03-aef8-d07a61136807')
        user.save()
        return user

    def update(self, instance, validated_data):
        instance.email = validated_data.get('email', instance.email)
        instance.first_name = validated_data.get('first_name', instance.first_name)
        instance.last_name = validated_data.get('last_name', instance.last_name)
        instance.default_tzid = validated_data.get('default_tzid', instance.default_tzid)

        instance.save()

        return instance

# class AppAuthorizationSerializer(serializers.ModelSerializer):
#
#     class Meta:
#         model = AppAuthorization
#         exclude = ('user',)


class ApplicationSerializer(serializers.ModelSerializer):

    class Meta:
        model = Application
        exclude = ('id', 'owner',)
        extra_kwargs = {'client_id': {'read_only': True},
                        'client_secret': {'read_only': True}}


class ApplicationUserSerializer(serializers.ModelSerializer):

    class Meta:
        model = ApplicationUser
        exclude = ('id',)
        extra_kwargs = {'application': {'read_only': True},
                        'user': {'read_only': True}}
