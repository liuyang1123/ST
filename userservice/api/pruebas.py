import requests
from api.utils import decode_token

token1 = "JWT eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjozNCwidXNlcm5hbWUiOiJhbGV4c3BlY3RlckBnbWFpbC5jb20iLCJvcmlnX2lhdCI6MTQ1NzkwMDAzMSwiZW1haWwiOiJhbGV4c3BlY3RlckBnbWFpbC5jb20iLCJleHAiOjE0NTgzMzIwMzF9.YhKymz2-cZ7My-DMp1lgsXP6BPCWnVHX_T6Z0PnbZ_A"
token2 = "Token eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjozNCwidXNlcm5hbWUiOiJhbGV4c3BlY3RlckBnbWFpbC5jb20iLCJvcmlnX2lhdCI6MTQ1NzkwMDAzMSwiZW1haWwiOiJhbGV4c3BlY3RlckBnbWFpbC5jb20iLCJleHAiOjE0NTgzMzIwMzF9.YhKymz2-cZ7My-DMp1lgsXP6BPCWnVHX_T6Z0PnbZ_A"
token3 = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjozNCwidXNlcm5hbWUiOiJhbGV4c3BlY3RlckBnbWFpbC5jb20iLCJvcmlnX2lhdCI6MTQ1NzkwMDAzMSwiZW1haWwiOiJhbGV4c3BlY3RlckBnbWFpbC5jb20iLCJleHAiOjE0NTgzMzIwMzF9.YhKymz2-cZ7My-DMp1lgsXP6BPCWnVHX_T6Z0PnbZ_A"
token4 = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjozNCwidXNlcm5hbWUiOiJhbGV4c3BlY3RlckBnbWFpbC5jb20iLCJvcmlnX2lhdCI6MTQ1NzkwMDAaaaWaaJhbGV4c3BlY3RlckBnbWFpbC5jb20iLCJleHAiOjE0NTgzMzIwMzF9.YhKymz2-cZ7My-DMp1lgsXP6BPCWnVHX_T6Z0PnbZ_A"
token5 = None

print("############################################")

print("")

print("=== DECODE TOKEN ===")
print(decode_token(token1))
print("")
print(decode_token(token2))
print("")
print(decode_token(token3))
print("")
print(decode_token(token4))
print("")
print(decode_token(token5))
print("")

request_token = None

print("=== REGISTER NEW USER - DEVELOPER ===")
try:
    request = requests.post(
        "http://127.0.0.1:8000/auth/register/",
        data={"email": "alexspecter@gmail.com",
              "first_name": "Alex", "last_name": "Specter",
              "default_tzid": "Europe/London",
              "password": "01598753"})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
    if "token" in request.json():
        request_token = request.json()['token']
    else:
        print("El resultado del request no contiene el token!")
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

print("")

print("=== VERIFY NEW USER - DEVELOPER TOKEN ===")
try:
    request = requests.post(
        "http://127.0.0.1:8000/auth/verify/",
        data={"token": request_token})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

print("")

request_token2 = None

print("=== GET TOKEN USER - DEVELOPER TOKEN ===")
try:
    request = requests.post(
        "http://127.0.0.1:8000/auth/token/",
        data={"email": "alexspecter@gmail.com",
              "password": "01598753"})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
    if "token" in request.json():
        request_token2 = request.json()['token']
    else:
        print("El resultado del request no contiene el token!")
    print("Los tokens son distintos?")
    print(request_token != request_token2)
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

print("")

refreshed_token = None

print("=== REFRESH TOKEN USER - DEVELOPER TOKEN ===")
try:
    request = requests.post(
        "http://127.0.0.1:8000/auth/refresh/",
        data={"token": request_token2})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
    refreshed_token = request.json()['token']
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

print("")

print("############################################")

print("")

new_application = None

print("=== CREATE APPLICATION ===")
try:
    request = requests.post(
        "http://127.0.0.1:8000/api/v1/applications/",
        headers={'Authorization': refreshed_token},
        data={"name": "RO", "url": "@RO"})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
    new_application = request.json()
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

print("")

print("=== LIST APPLICATIONS ===")
try:
    request = requests.get(
        "http://127.0.0.1:8000/api/v1/applications/",
        headers={'Authorization': refreshed_token})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

print("")

print("=== UPDATE APPLICATION ===")
try:
    request = requests.put(
        "http://127.0.0.1:8000/api/v1/applications/" +
        new_application['client_id'] + '/',
        headers={'Authorization': refreshed_token},
        data={"name": "#LetsGetReal", "url": "@LetsGetReal"})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

print("")

print("=== RETRIVE APPLICATION ===")
try:
    request = requests.get(
        "http://127.0.0.1:8000/api/v1/applications/" +
        new_application['client_id'] + '/',
        headers={'Authorization': refreshed_token})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

print("")

print("=== RETRIVE APPLICATION Not Owner ===")
try:
    request = requests.get(
        "http://127.0.0.1:8000/api/v1/applications/f06dfc67-725f-4e1c-8973-4caca53c5974/",
        headers={'Authorization': refreshed_token})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

print("")

print("############################################")

print("")

print("=== RETRIVE USER /me/ ===")
try:
    request = requests.get("http://127.0.0.1:8000/api/v1/users/me/",
                           headers={'Authorization': refreshed_token})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

print("")

print("=== UPDATE USER /me/ ===")
try:
    request = requests.put("http://127.0.0.1:8000/api/v1/users/me/",
                           data={"first_name": "Max",
                                 "default_tzid": "US/NYC"},
                           headers={'Authorization': refreshed_token})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
    if "token" in request.json():
        refreshed_token = request.json()['token']
    else:
        print("El resultado del request no contiene el token!")
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

print("")

print("=== RETRIVE USER AFTER UPDATE /me/ ===")
try:
    request = requests.get("http://127.0.0.1:8000/api/v1/users/me/",
                           headers={'Authorization': refreshed_token})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

print("")

print("############################################")

print("")

token_del_nuevo_user1 = None
token_del_nuevo_user2 = None
print("=== REGISTER NEW USER 1 - APP USER ===")
try:
    request = requests.post(
        "http://127.0.0.1:8000/api/v1/" +
        new_application['client_id'] + '/users/',
        data={"email": "testuser@gmail.com",
              "first_name": "Nope", "last_name": "Nope",
              "default_tzid": "Europe/London",
              "password": "01598753"})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
    if "token" in request.json():
        token_del_nuevo_user1 = request.json()['token']
    else:
        print("El resultado del request no contiene el token!")
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

print("")

print("=== REGISTER NEW USER 2 - APP USER ===")
try:
    request = requests.post(
        "http://127.0.0.1:8000/api/v1/" +
        new_application['client_id'] + '/users/',
        data={"email": "testuser1@gmail.com",
              "first_name": "Nope1", "last_name": "Nope1",
              "default_tzid": "Europe/London",
              "password": "01598753"})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
    if "token" in request.json():
        token_del_nuevo_user2 = request.json()['token']
    else:
        print("El resultado del request no contiene el token!")
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

print("")

print("=== VERIFY NEW USER - DEVELOPER TOKEN ===")
try:
    request = requests.post(
        "http://127.0.0.1:8000/auth/verify/",
        data={"token": token_del_nuevo_user1})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

print("")

user_data = None
print("=== LIST USERS ===")
try:
    request = requests.get("http://127.0.0.1:8000/api/v1/" + new_application['client_id'] + '/users/',
                           headers={'Authorization': refreshed_token})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
    user_data = request.json()
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

print("")

print("=== LIST USERS ===")
try:
    request = requests.get("http://127.0.0.1:8000/api/v1/" + new_application['client_id'] + '/users/',
                           headers={'Authorization': token_del_nuevo_user1})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

print("")

print("=== RETRIVE USERS OWNER ===")
try:
    request = requests.get(
        "http://127.0.0.1:8000/api/v1/" +
        new_application['client_id'] + '/users/' +
        user_data[0]['profile']['profile_id'] + '/',
        headers={'Authorization': refreshed_token})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

print("=== RETRIVE USERS SELF ===")
try:
    request = requests.get(
        "http://127.0.0.1:8000/api/v1/" +
        new_application['client_id'] + '/users/' +
        user_data[0]['profile']['profile_id'] + '/',
        headers={'Authorization': token_del_nuevo_user1})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

print("")

print("=== RETRIVE APPLICATION NOT Allowed ===")
try:
    request = requests.get(
        "http://127.0.0.1:8000/api/v1/" +
        new_application['client_id'] + '/users/' +
        user_data[0]['profile']['profile_id'] + '/',
        headers={'Authorization': token_del_nuevo_user2})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)


print("=== DELETE USERS ===")
try:
    request = requests.delete("http://127.0.0.1:8000/api/v1/" + new_application['client_id'] + '/users/' + user_data[1]['profile']['profile_id'] + '/',
                              headers={'Authorization': token_del_nuevo_user1})
    print("- Status code -")
    print(request.status_code)
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

print("")

print("=== UPDATE USERS ===")
try:
    request = requests.patch("http://127.0.0.1:8000/api/v1/" + new_application['client_id'] + '/users/' + user_data[0]['profile']['profile_id'] + '/',
                             headers={'Authorization': token_del_nuevo_user2},
                             data={"first_name": "656464343", })
    print("- Status code -")
    print(request.status_code)
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

print("")


print("=== RETRIVE PROFILES OWNER ===")
try:
    request = requests.get("http://127.0.0.1:8000/api/v1/" + new_application['client_id'] + '/profiles/' + user_data[0]['profile']['profile_id'] + '/',
                           headers={'Authorization': refreshed_token})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

print("=== RETRIVE PROFILES SELF ===")
try:
    request = requests.get(
        "http://127.0.0.1:8000/api/v1/" +
        new_application['client_id'] + '/profiles/' +
        user_data[0]['profile']['profile_id'] + '/',
        headers={'Authorization': token_del_nuevo_user2})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

print("")

print("=== UPDATE PROFILE ===")
try:
    request = requests.patch("http://127.0.0.1:8000/api/v1/" + new_application['client_id'] + '/profiles/' + user_data[0]['profile']['profile_id'] + '/',
                             headers={'Authorization': token_del_nuevo_user2},
                             data={"bio": "qwertyuiasdfgh", })
    print("- Status code -")
    print(request.status_code)
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

print("")

print("=== DELETE USER /me/ ===")
try:
    request = requests.delete("http://127.0.0.1:8000/api/v1/users/me/",
                              headers={'Authorization': refreshed_token})
    print("- Status code -")
    print(request.status_code)
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

print("")

print("=== DELETE APPLICATION ===")
try:
    request = requests.delete(
        "http://127.0.0.1:8000/api/v1/applications/" +
        new_application['client_id'] + '/',
        headers={'Authorization': refreshed_token})
    print("- Status code -")
    print(request.status_code)
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

print("")
