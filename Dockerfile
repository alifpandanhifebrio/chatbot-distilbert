FROM python:3.10-bookworm

ENV PYTHONUNBUFFERED True

ENV APP_HOME /back-end
WORKDIR $APP_HOME
COPY . ./

RUN pip install --no-cache-dir --upgrade pip

ENV PORT 80
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app

