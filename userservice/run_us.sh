#!/bin/bash
echo "[run] migrate"
python ./userservice/manage.py makemigrations api
python ./userservice/manage.py migrate

echo "[run] create superuser"
echo "from api.models import MyUser
if not MyUser.objects.filter(email='admin@nabulabs.com').count():
    MyUser.objects.create_superuser('admin@nabulabs.com', '01598753')
" | python ./userservice/manage.py shell

echo "[run] runserver"
python ./userservice/manage.py runserver 0.0.0.0:8000
