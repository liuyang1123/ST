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
                    headers = {'Authorization': request_token},
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
                    headers = {'Authorization': request_token})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

token_del_nuevo_user1 = None
token_del_nuevo_user2 = None
print("=== REGISTER NEW USER 1 - APP USER ===")
try:
    request = requests.post(
                    "http://127.0.0.1:8000/api/v1/" + new_application['client_id'] + '/users/',
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
                    "http://127.0.0.1:8000/api/v1/" + new_application['client_id'] + '/users/',
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

user_data = None
print("=== LIST USERS ===")
try:
    request = requests.get("http://127.0.0.1:8000/api/v1/" + new_application['client_id'] + '/users/',
                           headers = {'Authorization': request_token})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
    user_data = request.json()
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

calendar_id = None

print("=== CREATE NEW CALENDAR ===")
try:
    request = requests.post("http://127.0.0.1:8001/api/v1/calendars/",
                           headers = {'Authorization': token_del_nuevo_user1},
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

print("=== LIST CALENDARS ===")
try:
    request = requests.get("http://127.0.0.1:8001/api/v1/calendars/",
                           headers = {'Authorization': token_del_nuevo_user1})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

print("=== RETRIVE CALENDAR ===")
try:
    request = requests.get("http://127.0.0.1:8001/api/v1/calendars/" + calendar_id + '/',
                           headers = {'Authorization': token_del_nuevo_user1})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

print("=== UPDATE CALENDAR ===")
try:
    request = requests.put("http://127.0.0.1:8001/api/v1/calendars/" + calendar_id + '/',
                           headers = {'Authorization': token_del_nuevo_user1},
                           data={"provider_name": "apple",
                                 "calendar_name": "Home"})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)











print("=== CREATE NEW EVENT ===")
try:
    request = requests.post("http://127.0.0.1:8001/api/v1/" + calendar_id + '/events/',
                           headers = {'Authorization': token_del_nuevo_user1},
                           data={"location": "Board room",
                                 "attendees": ["alexspecter@nabulabs.com", "a2@gmail.com"]})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
    calendar_id = request.json()['data']['generated_keys'][0]
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)


f = open("error.html", "w")
f.write(request.text)
f.close()






print("=== LIST EVENTS ===")
try:
    request = requests.get("http://127.0.0.1:8001/api/v1/" + calendar_id + '/events/',
                           headers = {'Authorization': token_del_nuevo_user1})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)



print("=== RETRIVE EVENT ===")
try:
    request = requests.get("http://127.0.0.1:8001/api/v1/calendars/" + calendar_id + '/',
                           headers = {'Authorization': token_del_nuevo_user1})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

print("=== UPDATE EVENT ===")
try:
    request = requests.put("http://127.0.0.1:8001/api/v1/calendars/" + calendar_id + '/',
                           headers = {'Authorization': token_del_nuevo_user1},
                           data={"provider_name": "apple",
                                 "calendar_name": "Home"})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

print("=== DELETE EVENT ===")
try:
    request = requests.delete("http://127.0.0.1:8001/api/v1/calendars/" + calendar_id + '/',
                           headers = {'Authorization': token_del_nuevo_user1})
    print("- Status code -")
    print(request.status_code)
    print("- JSON -")
    print(request.json())
except requests.exceptions.RequestException as e:
    print("- ERROR detected -")
    print(e)

























print("=== DELETE CALENDAR ===")
try:
    request = requests.delete("http://127.0.0.1:8001/api/v1/calendars/" + calendar_id + '/',
                           headers = {'Authorization': token_del_nuevo_user1})
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
