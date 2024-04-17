import os

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from inference_pystore import main
from weekly_grid_search import main_grid_search

# set daily start time
HOUR = 15
MIN = 4
DAY = 2 # 0-6

def log_job():
    main()
    main_grid_search()

if __name__ == '__main__':
    sched = BlockingScheduler()
    sched.add_job(log_job, 'cron', day_of_week=DAY, hour=HOUR,minute=MIN)
    sched.start()


