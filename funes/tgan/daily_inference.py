import os
from apscheduler.schedulers.blocking import BlockingScheduler

# set daily start time
HOUR = 17
MIN = 00

def log_job():
    os.execlp(
        "python",
        "python",
        "inference.py"
    )  #  run script

if __name__ == '__main__':
    sched = BlockingScheduler()
    sched.add_job(log_job, 'cron', hour=HOUR, minute=MIN)
    sched.start()
