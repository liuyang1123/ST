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
