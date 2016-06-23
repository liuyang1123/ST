"""userservice URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.9/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url, include
from django.contrib import admin
from rest_framework_jwt.views import obtain_jwt_token, verify_jwt_token, refresh_jwt_token
from django.views.generic import RedirectView
from api.views import router, RegistrationView, UserView, AllUsersView


urlpatterns = [
    url(r'^admin/', admin.site.urls),

    url(r'^auth/register/', RegistrationView.as_view()),
    url(r'^auth/token/', obtain_jwt_token),
    url(r'^auth/refresh/', refresh_jwt_token),
    url(r'^auth/verify/', verify_jwt_token),

    url(r'^api/v1/users/me/', UserView.as_view()),
    url(r'^api/v1/list_all_users/', AllUsersView.as_view()),
    url(r'^api/v1/', include(router.urls)),

    url(r'^api-explorer/', include('rest_framework_swagger.urls')),
    url(r'^', RedirectView.as_view(url='/api-explorer/')),
]
