#!/bin/sh

cd msservice
su -m myuser -c "celery worker -A msservice.celeryconf -Q default -n default@%h"
