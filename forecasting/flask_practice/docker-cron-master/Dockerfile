FROM ubuntu:latest

# Install cron
RUN apt-get update
RUN apt-get install cron
RUN apt-get install sudo
RUN apt-get -y install python3
#ADD apt-repository ppa:jonathonf/python3.6
RUN apt-get -y install python3-pandas

# Add crontab file in the cron directory
ADD crontab /etc/cron.d/simple-cron

# Add shell script and grant execution rights

#ADD script.sh /script.sh
ADD test.py /test.py

#RUN chmod +x /script.sh
RUN chmod +x /test.py

# Give execution rights on the cron job
RUN chmod 0644 /etc/cron.d/simple-cron

# Create the log file to be able to run tail
RUN touch /var/log/cron.log

# Run the command on container startup
CMD cron && tail -f /var/log/cron.log
