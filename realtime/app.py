#!/usr/bin/env python
from dateutil.parser import parse
from flask import Flask, render_template, session, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room, \
    close_room, rooms, disconnect
import json
import uuid
import requests
from nl_manager import ConversationManager
from witai import Wit

WIT_ACCESS_TOKEN = 'N3YY33JVMGXRJG7GKY3LA4BXUVXIFYCK'

# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on installed packages.
async_mode = None

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ec3f9d37-0004-4798-81ac-443237317bb6'
socketio = SocketIO(app, async_mode=async_mode)
thread = None


class ChatSession(object):

    def __init__(self, session, msg, user=None, attributes={}):
        self.session = session
        self.user = user
        self.message = msg
        self.attributes = attributes

    def save(self):
        j = self.to_dict()
        cm = ConversationManager()
        inserted = cm.insert(j)
        cm.close()
        # save in rethinkdb

    def to_dict(self):
        return {"session": self.session,
                "user": self.user,
                "message": self.message,
                "attributes": self.attributes}


@app.route('/')
def index():
    return render_template('index.html', async_mode=socketio.async_mode)


@socketio.on('connect', namespace='/wb')
def on_connect():
    print("Connected")
    # emit('my response', {'data': 'Connected', 'count': 0})


@socketio.on('disconnect', namespace='/wb')
def on_disconnect():
    print("Disconnected")


@socketio.on('disconnect request', namespace='/wb')
def on_disconnect_request():
    print("Disconnect request")
    disconnect()


@socketio.on('join', namespace='/wb')
def join(message={}):
    r_id = message.get('room', str(uuid.uuid4()))
    join_room(r_id)
    emit('joined', {'room': r_id})


@socketio.on('leave', namespace='/wb')
def leave(message):
    # leave_room(message['room'])
    close_room(message['room'])


def send(request, response):
    cs = ChatSession(session=request['session_id'],
                     msg=response['text'])
    cs.save()
    emit('response', {'message': response['text']}, room=request['session_id'])


def first_entity_value(entities, entity):
    if entity not in entities:
        return None
    val = entities[entity][0].get('value', entities[entity][0])
    if not val:
        return None
    return val.get('value', val) if isinstance(val, dict) else val


def first_entity_value_time(entities, entity):
    if entity not in entities:
        return None, None
    val = entities[entity][0].get('value', entities[entity][0])
    x1 = parse(val["values"][0]["from"]["value"]).time().isoformat()
    x2 = parse(val["values"][0]["to"]["value"]).time().isoformat()

    return x1, x2


def schedule_event(request):
    print("schedule_event")
    print(request)
    context = request['context']
    entities = request['entities']

    # TODO Call SS to start the scheduling algorithm
    #
    # loc = first_entity_value(entities, 'location')

    event_type = first_entity_value(entities, 'event_type')
    duration = first_entity_value(entities, 'duration')
    location = first_entity_value(entities, 'location')
    start = first_entity_value(entities, 'datetime')
    try:
        request = requests.post(
            "http://127.0.0.1:9000/api/v1/scheduling/",
            headers={'Authorization': request.get('token')},
            data={"duration": duration,
                  "location": location,
                  "event_type": event_type,
                  "start": start,
                  "end": None,
                  "attendees": [],
                  "calendar_id": -1})
        context['timeslot'] = request.json()['start']
    except requests.exceptions.RequestException as e:
        print("- ERROR detected -")
        print(e)

    return context

    # Call the SS Api to start the scheduling job
    # return request['context']


def infer_attendance(request):
    print("infer_attendance")
    print(request)
    context = request['context']
    entities = request['entities']

    # TODO Call SS to start the scheduling algorithm
    #
    # loc = first_entity_value(entities, 'location')

    event_type = first_entity_value(entities, 'event_type')
    duration = first_entity_value(entities, 'duration')
    obs = json.dumps({"type": event_type, "duration": duration})
    try:
        request = requests.post(
            "http://127.0.0.1:9000/api/v1/dpreferences/infer/",
            headers={'Authorization': request.get('token')},
            data={'observation': obs})
        context['result'] = request.json()
    except requests.exceptions.RequestException as e:
        print("- ERROR detected -")
        print(e)

    return context


def update_preference(request):
    print("update_preference")
    print(request)
    context = request['context']
    entities = request['entities']

    d = {}
    a = {}
    event_type = first_entity_value(entities, 'event_type')
    d["event_type"] = event_type
    duration = first_entity_value(entities, 'duration')
    if duration is not None:
        a["duration"] = duration
    preference_type = first_entity_value(entities, 'preference_type')
    d["preference_type"] = preference_type
    _start_time, _end_time = first_entity_value_time(entities, 'datetime')
    if _start_time is not None:
        a["start_time"] = _start_time
        a["end_time"] = _end_time

    #location = first_entity_value(entities, 'duration')
    attributes = json.dumps(a)
    d["attributes"] = attributes
    try:
        request = requests.post(
            "http://127.0.0.1:9000/api/v1/preferences/",
            headers={'Authorization': request.get('token')},
            data=d)
        context['result'] = request.json()
    except requests.exceptions.RequestException as e:
        print("- ERROR detected -")
        print(e)

    return context

ACTIONS = {
    'send': send,
    'scheduleEvent': schedule_event,
    'inferAttendance': infer_attendance,
    'updatePreference': update_preference
}

client = Wit(access_token=WIT_ACCESS_TOKEN, actions=ACTIONS)


@socketio.on('message', namespace='/wb')
def message(message):
    print("Incoming")
    print(message)
    # Incoming message
    cs = ChatSession(session=message.get('room'),
                     msg=message.get('message'),
                     user=1)
    cs.save()
    d = client.run_actions(session_id=message.get('room'),
                           message=message.get('message'),
                           user_token=message.get('token'))


@app.route('/events', methods=['GET'])
def get_tasks():
    return jsonify({'status': 'ok'})


# TODO Change the app setting the default channel id as the user_id
# @app.route('/invitation', methods=['POST'])
# def send_invitation(invitation):
#     emit('response', {'message': invitation},
#          room=invitation.user)


# def changes():
#     conn = r.connect( "localhost", 28015)
#     feed = r.db('calendarapi').table('events').changes().run(conn)
#     for change in feed:
#         print("new change")
#         print(change)
#         socketio.emit('message', change, namespace='/test')
#
# @socketio.on('listen-changes', namespace='/test')
# def listen_changes(message):
#     t = socketio.start_background_task(target=listen_for_changes)

if __name__ == '__main__':
    socketio.run(app, debug=True)
