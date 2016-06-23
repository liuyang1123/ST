import json
from django.test import TestCase, Client
from django.core.urlresolvers import reverse
from rest_framework_jwt.settings import api_settings
from api.models import MyUser, Profile


class TokenAuthTestCase(TestCase):

    def setUp(self):
        self.user = MyUser.objects.create_user(email="joe@gmail.com", password="test")
        self.c = Client()

    def test_get_token(self):

        response = self.c.post(
            "/api/v1/api-token-auth/", {"email": "joe@gmail.com", "password": "test"})

        assert 200 == response.status_code, "Response is 200 OK"
        assert json.loads(response.content.decode('utf-8')).get("token", False) is not False, "Token is set"


class ModelsTestCase(TestCase):

    def test_new_user_signals(self):

        user = MyUser.objects.create_user(email="joe@gmail.com", password="test")
        
        assert Profile.objects.filter(
            user=user).count() == 1, "Exactly 1 matching profile is created"


class EndpointAuthenticationTestCase(TestCase):

    def setUp(self):
        self.c = Client()
        self.user = MyUser.objects.create_user(email="joe@gmail.com", password="test")

    def test_endpoints_require_token_auth(self):
        url = reverse('myuser-me')
        response = self.c.get(url)

        assert response.status_code == 403, "403: Authentication required"

    def test_user_can_see_self(self):
        url = reverse('myuser-me')
        jwt_payload_handler = api_settings.JWT_PAYLOAD_HANDLER
        jwt_encode_handler = api_settings.JWT_ENCODE_HANDLER
        payload = jwt_payload_handler(self.user)
        token = jwt_encode_handler(payload)

        response = self.c.get(
            url, HTTP_AUTHORIZATION="JWT {0}" . format(token))

        assert response.status_code == 200, "User can see self"