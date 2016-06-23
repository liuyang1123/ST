#!/bin/bash
echo "[run] migrate"
python ./msservice/manage.py makemigrations api
python ./msservice/manage.py migrate
echo "from api.event_module.manager import run_setup
run_setup()
" | python ./msservice/manage.py shell
echo "from api.learning_module.hard_constraints.preferences.manager import run_setup
run_setup()
" | python ./msservice/manage.py shell

echo "[run] runserver"
python ./msservice/manage.py runserver 0.0.0.0:9000
