import os

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler


# from sine_grid_search import sine_grid_search

from mixed_data_grid_search import mixed_data_grid_search

# from grid_search_test import main_grid_search

# set daily start time
HOUR = 18
MIN = 5
DAY = 6 # 0-6

def log_job():
    mixed_data_grid_search()

if __name__ == '__main__':
    sched = BlockingScheduler()
    sched.add_job(log_job, 'date', run_date='2022-12-17 23:00:00')
    sched.start()

