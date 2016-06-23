#!/bin/bash
echo "[run] setup database"
echo "from api.models import run_setup
run_setup()
" | python ./calendarservice/manage.py shell

echo "[run] runserver"
python ./calendarservice/manage.py runserver 0.0.0.0:8001
