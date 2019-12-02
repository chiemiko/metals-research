FROM Ubuntu:latest

# Linux commands
RUN apt-get update
RUN apt-get install python
RUN apt-get -y install cron

#python:3.7.3 

#WORKDIR /app
#RUN pip install -r requirements.txt
#CMD ["gunicorn", "-w 4", "main:app"]

# ENTRYPOINT ["echo", "hello world"]


RUN pip install flask
RUN pip install flask-mysql


COPY . /opt/source-code

ENTRYPOINT FLASK_APP = /opt/source-code/app.py flask run
