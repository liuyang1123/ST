from rest_framework.permissions import BasePermission
from api.utils import decode_token


class IsAuthenticated(BasePermission):
    """
    Allows access only to authenticated users.
    """

    def has_permission(self, request, view):
        token = decode_token(request.META)
        if token is None:
            return False
        return True

SAFE_METHODS = ['POST']


class UserViewSetPermissions(BasePermission):
    """
    Allows access only to authenticated users.
    """

    def has_object_permission(self, request, view, obj):
        # Write permissions are only allowed to the owner of the snippet.
        if request.method not in SAFE_METHODS:
            token = decode_token(request.META)
            return obj.email == token['email']
        return True

    def has_permission(self, request, view):
        if request.method in SAFE_METHODS:
            return True
        token = decode_token(request.META)

        if token is None:
            return False
        return True


class ProfileViewSetPermissions(BasePermission):
    """
    Allows access only to authenticated users.
    """

    def has_object_permission(self, request, view, obj):
        # Write permissions are only allowed to the owner of the snippet.
        token = decode_token(request.META)
        return obj.user.email == token['email']

    def has_permission(self, request, view):
        token = decode_token(request.META)

        if token is None:
            return False
        return True
