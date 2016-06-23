import requests

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

new_application = None

print("=== CREATE APPLICATION ===")
try:
    request = requests.post(
        "http://127.0.0.1:8000/api/v1/applications/",
        headers={'Authorization': request_token},
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

token_del_nuevo_user1 = None
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

# preference_id = None
# print("=== CREATE NEW PREFERENCE ===")
# try:
#     request = requests.post("http://127.0.0.1:9000/api/v1/preferences/",
#                             headers={'Authorization': token_del_nuevo_user1},
#                             data={"preference_name": "timebetweenpreference",
#                                   "event_type": "any",
#                                   "attributes": b'{"initial_time": 1, "final_time": 2}'})
#     print("- Status code -")
#     print(request.status_code)
#     print("- JSON -")
#     print(request.json())
#     preference_id = request.json()['data']['generated_keys'][0]
# except requests.exceptions.RequestException as e:
#     print("- ERROR detected -")
#     print(e)
#
#
# print("=== LIST PREFERENCES ===")
# try:
#     request = requests.get("http://127.0.0.1:9000/api/v1/preferences/",
#                            headers={'Authorization': token_del_nuevo_user1})
#     print("- Status code -")
#     print(request.status_code)
#     print("- JSON -")
#     print(request.json())
# except requests.exceptions.RequestException as e:
#     print("- ERROR detected -")
#     print(e)
#
# print("=== RETRIVE PREFERENCE ===")
# try:
#     request = requests.get("http://127.0.0.1:9000/api/v1/preferences/" + preference_id + '/',
#                            headers={'Authorization': token_del_nuevo_user1})
#     print("- Status code -")
#     print(request.status_code)
#     print("- JSON -")
#     print(request.json())
# except requests.exceptions.RequestException as e:
#     print("- ERROR detected -")
#     print(e)
#
# print("=== UPDATE PREFERENCE ===")
# try:
#     request = requests.put("http://127.0.0.1:9000/api/v1/preferences/" + preference_id + '/',
#                            headers={'Authorization': token_del_nuevo_user1},
#                            data={"preference_name": "distancelocationpreference",
#                                  "attributes": b'{"duration": 10}'})
#     print("- Status code -")
#     print(request.status_code)
#     print("- JSON -")
#     print(request.json())
# except requests.exceptions.RequestException as e:
#     print("- ERROR detected -")
#     print(e)
#
# print("=== DELETE PREFERENCE ===")
# try:
#     request = requests.delete("http://127.0.0.1:9000/api/v1/preferences/" + preference_id + '/',
#                               headers={'Authorization': token_del_nuevo_user1})
#     print("- Status code -")
#     print(request.status_code)
#     print("- JSON -")
#     print(request.json())
# except requests.exceptions.RequestException as e:
#     print("- ERROR detected -")
#     print(e)
#
# print("=== CREATE DEFAULT PREFERENCES ===")
# try:
#     request = requests.post("http://127.0.0.1:9000/api/v1/preferences/default/",
#                             headers={'Authorization': token_del_nuevo_user1})
#     print("- Status code -")
#     print(request.status_code)
#     print("- JSON -")
#     print(request.json())
# except requests.exceptions.RequestException as e:
#     print("- ERROR detected -")
#     print(e)


calendar_id = None

print("=== CREATE NEW CALENDAR ===")
try:
    request = requests.post("http://127.0.0.1:8001/api/v1/calendars/",
                            headers={'Authorization': token_del_nuevo_user1},
                            data={"provider_name": "google",
                                  "calendar_name": "Work"})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
    calendar_id = request.json()['data']['generated_keys'][0]
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

print("=== CREATE NEW EVENT ===")
try:
    request = requests.post("http://127.0.0.1:9000/api/v1/scheduling/",
                            headers={'Authorization': token_del_nuevo_user1},
                            data={"calendar_id": calendar_id,
                                  "location": "Board room",
                                  "attendees": ["alexspecter@nabulabs.com", "a2@gmail.com"]})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

print("=== LIST TASKS ===")
try:
    request = requests.get("http://127.0.0.1:9000/api/v1/scheduling/",
                           headers={'Authorization': token_del_nuevo_user1})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)


f = open("error.html", "w")
f.write(request.text)
f.close()
