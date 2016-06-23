import requests
from rest_framework_jwt.settings import api_settings
from calendarservice import settings

def verify_token(token):
    try:
        request = requests.post(settings.USER_SERVICE_TOKEN_VERIFICATION_URL,
                                data={"token": str(token)})

        if request.status_code == 200:
            return True
    except requests.exceptions.RequestException as e:
        pass
    return False


def decode_token(data):
    token = data.get('HTTP_AUTHORIZATION', None)

    if token is None:
        return None

    token = token.split(' ') # [JWT,XXXX] or [Token,XXXX]
    if len(token) > 1: # If not: token = ['XXXX']
        token = token[1]
    else:
        token = token[0]

    if verify_token(token):
        jwt_decode_handler = api_settings.JWT_DECODE_HANDLER
        decoded_data = jwt_decode_handler(token)

        return decoded_data

    return None
