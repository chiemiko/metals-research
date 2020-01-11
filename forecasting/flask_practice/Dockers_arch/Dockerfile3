# From
# https://serverfault.com/questions/136461/how-to-check-cron-logs-in-ubuntu
#

FROM alpine:3.6

COPY config/cronjobs /etc/crontabs/root

CMD ["crond","-f","-d","8"]
