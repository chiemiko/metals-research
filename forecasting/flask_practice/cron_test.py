# FOR JINJA 
# https://www.youtube.com/watch?v=xh3mFxbnc4o

from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask

def sensor():
    """ Function for test purposes. """
    print("Scheduler is alive!")

sched = BackgroundScheduler(daemon=True)
sched.add_job(sensor,'interval',minutes=60)
sched.start()

app = Flask(__name__)

@app.route("/home")
def home():
    """ Function for test purposes. """
    return "Welcome Home :) !"

if __name__ == "__main__":
    app.run()